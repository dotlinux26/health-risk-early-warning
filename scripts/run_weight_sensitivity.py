"""P0.4 — Độ nhạy của risk score theo cấu hình trọng số (docs/15 mục 5.4).

Chạy pipeline đầy đủ (3 tầng) cho từng bệnh nhân mẫu dưới các bộ trọng số:

    v1 = 0.30 / 0.35 / 0.25 / 0.10   (hiện hành)
    v2 = 0.25 / 0.40 / 0.25 / 0.10
    v3 = 0.30 / 0.30 / 0.30 / 0.10

Báo cáo: phân bố mức nguy cơ, mức đồng thuận risk-level giữa các bộ, các chuyển
đổi Low→Medium / Medium→High, phương vị điểm.

Nguyên tắc: KHÔNG tối ưu trọng số để "đẹp số" khi chưa có ground-truth outcome.
Nếu kết quả nhạy với trọng số → ghi nhận: "risk score nhạy với cấu hình trọng
số; các trọng số hiện tại chỉ là thiết kế ban đầu".

Chạy:
    python scripts/run_weight_sensitivity.py [--data data/sample_long.csv]

Output:
    experiments/WEIGHT-SENSITIVITY/{summary.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402
from src.core.pipeline import assess_patient  # noqa: E402
from src.tier3_risk.scoring import RiskScorer  # noqa: E402

OUT_DIR = Path("experiments/WEIGHT-SENSITIVITY")
WEIGHT_SETS = {
    "v1_30_35_25_10": {"stat": 0.30, "knowledge": 0.35, "ml": 0.25, "trend": 0.10},
    "v2_25_40_25_10": {"stat": 0.25, "knowledge": 0.40, "ml": 0.25, "trend": 0.10},
    "v3_30_30_30_10": {"stat": 0.30, "knowledge": 0.30, "ml": 0.30, "trend": 0.10},
}


def load_wide(path: Path) -> dict[str, pd.DataFrame]:
    """Long CSV -> {patient_id: DataFrame wide (index ngày, cột metric)}."""
    long = pd.read_csv(path, parse_dates=["timestamp"])
    out = {}
    for pid, g in long.groupby("patient_id"):
        w = g.pivot_table(index="timestamp", columns="metric",
                          values="value", aggfunc="mean")
        w = w.sort_index().reset_index()
        out[str(pid)] = w
    return out


NHANES_TO_SYSTEM = {
    "systolic_bp": ("systolic_bp", 6.0),
    "diastolic_bp": ("diastolic_bp", 4.0),
    "heart_rate": ("heart_rate", 5.0),
    "glucose_fasting": ("glucose_fasting", 0.45),
}


def augment_from_nhanes(n_extra: int, days: int = 14) -> dict[str, pd.DataFrame]:
    """Tạo bệnh nhân giả từ mẫu cắt ngang NHANES (giá trị riêng + noise).

    Mục đích: đủ mẫu để thống kê mức đồng thuận. Đây là dữ liệu tổng hợp,
    KHÔNG dùng cho báo cáo hiệu năng ML.
    """
    path = Path("data/datasets/nhanes_merged.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    rng = np.random.default_rng(7)
    pick = df.sample(min(n_extra, len(df)), random_state=7)
    out = {}
    ts = pd.date_range("2025-06-01", periods=days, freq="D")
    for i, (_, row) in enumerate(pick.iterrows()):
        cols = {"timestamp": ts}
        for src, (dst, sigma) in NHANES_TO_SYSTEM.items():
            base = row.get(src)
            if base is None or pd.isna(base):
                cols[dst] = np.full(days, np.nan) if dst == "glucose_fasting" else np.full(days, 75.0 if dst == "heart_rate" else np.nan)
                continue
            series = float(base) + rng.normal(0, sigma, days)
            if dst == "glucose_fasting":
                series[rng.random(days) < 0.3] = np.nan
            cols[dst] = series
        out[f"SYN{i:04d}"] = pd.DataFrame(cols)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Weight sensitivity P0.4")
    parser.add_argument("--data", type=str, default="data/sample_long.csv")
    parser.add_argument("--augment", type=int, default=150,
                        help="Số bệnh nhân giả bổ sung từ NHANES (0 = tắt)")
    args = parser.parse_args()

    patients = load_wide(Path(args.data))
    n_real = len(patients)
    if args.augment > 0:
        patients.update(augment_from_nhanes(args.augment))
    print(f"Bệnh nhân: {n_real} mẫu thật + {len(patients) - n_real} giả từ NHANES")

    results: dict[str, dict] = {}
    scores_by_version: dict[str, dict[str, float]] = {}
    levels_by_version: dict[str, dict[str, str]] = {}
    for vname, weights in WEIGHT_SETS.items():
        cfg = Config()
        cfg.risk_weights = dict(weights)
        scorer = RiskScorer(cfg)
        lvls, scs = {}, {}
        for pid, wide in patients.items():
            try:
                res = assess_patient(wide.copy(), cfg, scorer)
                if res.get("risk_level") == "INSUFFICIENT_DATA":
                    continue
                lvls[pid] = res["risk_level"]
                scs[pid] = float(res.get("risk_score", 0.0))
            except Exception as e:  # noqa: BLE001 — ghi nhận và đi tiếp
                print(f"  [{vname}] lỗi {pid}: {e}")
        levels_by_version[vname] = lvls
        scores_by_version[vname] = scs
        dist = Counter(lvls.values())
        results[vname] = {
            "weights": weights,
            "n_assessed": len(lvls),
            "level_distribution": dict(dist),
            "score_mean": round(float(np.mean(list(scs.values()))), 4),
            "score_std": round(float(np.std(list(scs.values()))), 4),
        }
        print(f"  {vname}: {dict(dist)} | score mean={results[vname]['score_mean']}"
              f"±{results[vname]['score_std']}")

    # So sánh từng cặp phiên bản
    pair_stats = {}
    for a, b in combinations(WEIGHT_SETS, 2):
        pids = set(levels_by_version[a]) & set(levels_by_version[b])
        if not pids:
            continue
        same = sum(levels_by_version[a][p] == levels_by_version[b][p] for p in pids)
        trans = Counter()
        for p in pids:
            la, lb = levels_by_version[a][p], levels_by_version[b][p]
            order = ["THAP", "TRUNG_BINH", "CAO"]
            la_i, lb_i = order.index(la), order.index(lb)
            if la_i < lb_i:
                trans[f"{la}→{lb}"] += 1
            elif la_i > lb_i:
                trans[f"{la}→{lb}"] += 1
        sd = np.std([scores_by_version[a][p] - scores_by_version[b][p]
                     for p in pids])
        pair_stats[f"{a}|{b}"] = {
            "n_common": len(pids),
            "level_agreement": round(same / len(pids), 4),
            "transitions": dict(trans),
            "score_diff_std": round(float(sd), 4),
        }
        print(f"  {a} vs {b}: đồng thuận {same}/{len(pids)} = {same/len(pids):.1%} "
              f"| chuyển mức: {dict(trans)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": args.data, "versions": results, "pairs": pair_stats}
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# P0.4 Độ nhạy risk score theo bộ trọng số",
        "",
        f"- Dữ liệu: `{args.data}` ({n_real} bệnh nhân thật) + "
        f"{len(patients) - n_real} bệnh nhân giả từ NHANES (pipeline 3 tầng đầy đủ).",
        "- Lưu ý an toàn: luật severity ≥ 0.7 vẫn kích hoạt sàn TRUNG_BÌNH ở mọi bộ.",
        "",
        "| Bộ trọng số | Phân bố mức | Score mean±std |",
        "|---|---|---|",
    ]
    label = {"v1_30_35_25_10": "v1 (hiện hành)", "v2_25_40_25_10": "v2",
             "v3_30_30_30_10": "v3"}
    for v, r in results.items():
        md.append(f"| {label.get(v, v)} | {r['level_distribution']} "
                  f"| {r['score_mean']:.3f}±{r['score_std']:.3f} |")
    md += [
        "",
        "| Cặp so sánh | Đồng thuận mức | Chuyển mức | std(Δscore) |",
        "|---|---|---|---|",
    ]
    for k, s in pair_stats.items():
        a, b = k.split("|")
        md.append(f"| {label.get(a, a)} vs {label.get(b, b)} "
                  f"| {s['level_agreement']:.1%} | {s['transitions'] or '—'} "
                  f"| {s['score_diff_std']:.3f} |")
    md += [
        "",
        "> Kết luận áp dụng: nếu đồng thuận cao (≥95%) → risk level bền vững với",
        "> thiết kế trọng số. Nếu thấp → risk score nhạy với trọng số; các trọng",
        "> số hiện tại chỉ là thiết kế ban đầu và phải được ghi rõ trong báo cáo.",
        "> Không hiệu chỉnh trọng số nhằm cải thiện số khi chưa có outcome thật.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nKết quả lưu tại: {OUT_DIR}/")


if __name__ == "__main__":
    main()
