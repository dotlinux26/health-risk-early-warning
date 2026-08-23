"""P0.2 — Độ nhạy nhãn (label sensitivity) theo docs/15 mục 5.2.

So sánh hai cách định nghĩa nhãn trên CÙNG protocol, CÙNG test set:

    A: label        = HTN(sbp>=140 | dbp>=90 | đang dùng thuốc HA) | DM(hba1c>=6.5 | fpg>=7.0)
    B: label_no_med = HTN(sbp>=140 | dbp>=90 — bỏ thành phần thuốc)  | DM(giữ nguyên)

Cùng split (phân tầng theo nhãn A), cùng mô hình, cùng seed.
Nếu A ~ B -> thành phần thuốc không làm kết quả đổi lớn.
Nếu khác  -> bằng chứng nhãn nhạy với cách định nghĩa (cũng là kết quả nghiên cứu).

Lưu ý: dù A ~ B, vòng lặp nhãn–đầu vào (D1) KHÔNG biến mất — model vẫn chủ yếu
tái lập phân tầng theo tiêu chí lâm sàng chứ chưa dự báo biến cố tương lai.

Chạy:
    python scripts/run_label_sensitivity.py                  # lgbm,xgb,lr x 5 seed
    python scripts/run_label_sensitivity.py --models lgbm

Output:
    experiments/LABEL-SENSITIVITY/{summary.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.models import available_models  # noqa: E402
from src.experiments.protocol import (  # noqa: E402
    NHANES_FEATURES,
    SEEDS,
    ExperimentConfig,
    evaluate_metrics,
    stratify_split,
)

OUT_DIR = Path("experiments/LABEL-SENSITIVITY")
DEFAULT_MODELS = ["lgbm", "xgb", "lr"]


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Nhãn A (giữ nguyên 'label', có thành phần thuốc HA) và B (bỏ thuốc)."""
    htn_nomeds = (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
    dm = (df["hba1c"] >= 6.5) | (df["glucose_fasting"] >= 7.0)
    df = df.copy()
    df["label_A"] = df["label"].astype(int)
    df["label_B"] = (htn_nomeds | dm).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Label sensitivity A vs B")
    parser.add_argument("--data", type=str, default="data/datasets/nhanes_merged.csv")
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS, choices=available_models()
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["label"])
    df = build_labels(df)
    features = list(NHANES_FEATURES)
    X = df[features].copy()

    pos_a, pos_b = df["label_A"], df["label_B"]
    overlap = {
        "n": int(len(df)),
        "positive_rate_A": round(float(pos_a.mean()), 4),
        "positive_rate_B": round(float(pos_b.mean()), 4),
        "positives_only_in_A": int(((pos_a == 1) & (pos_b == 0)).sum()),
        "positives_only_in_B": int(((pos_a == 0) & (pos_b == 1)).sum()),
        "changed_total": int((pos_a != pos_b).sum()),
    }
    print(f"Nhãn: {overlap}")

    rows: list[dict] = []
    for key in args.models:
        for seed in args.seeds:
            cfg = ExperimentConfig(
                model_key=key, seed=seed, dataset=args.data, features=features
            )
            # Split MỘT LẦN theo nhãn A -> cùng test set cho cả A và B.
            # LƯU Ý: train_test_split trả về subset theo thứ tự đã xáo trộn,
            # nên nhãn B PHẢI lấy bằng .loc[index] (không dùng mask .isin()
            # theo thứ tự gốc — sẽ làm lệch nhãn so với hàng).
            splits = stratify_split(X, pos_a.to_numpy(dtype=int), seed, cfg)
            X_tr, X_va, X_te, yA_tr, yA_va, yA_te = splits
            yB_tr = pos_b.loc[X_tr.index].to_numpy(dtype=int)
            yB_va = pos_b.loc[X_va.index].to_numpy(dtype=int)
            yB_te = pos_b.loc[X_te.index].to_numpy(dtype=int)

            imp = SimpleImputer(strategy="median")
            cols = list(X.columns)
            Xtr = pd.DataFrame(imp.fit_transform(X_tr), columns=cols)
            Xva = pd.DataFrame(imp.transform(X_va), columns=cols)
            Xte = pd.DataFrame(imp.transform(X_te), columns=cols)

            res = {"model": key, "seed": seed}
            for tag, y_tr_, y_va_, y_te_ in (
                ("A", yA_tr, yA_va, yA_te),
                ("B", yB_tr, yB_va, yB_te),
            ):
                from src.experiments.models import build_model

                model = build_model(key, seed)
                model.fit(Xtr, y_tr_)
                proba = model.predict_proba(Xte)[:, 1]
                m = evaluate_metrics(y_te_, proba)
                res.update({f"{k}_{tag}": v for k, v in m.items()})
            for k in ("roc_auc", "pr_auc", "brier", "f1"):
                res[f"delta_{k}"] = round(res[f"{k}_B"] - res[f"{k}_A"], 6)
            rows.append(res)
            print(
                f"  {cfg.experiment_id}: AUC A={res['roc_auc_A']:.4f} "
                f"B={res['roc_auc_B']:.4f} d={res['delta_roc_auc']:+.4f}"
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": args.data, "labels_overlap": overlap, "runs": rows}

    def agg(model: str, suffix: str) -> dict[str, tuple[float, float]]:
        sub = [r for r in rows if r["model"] == model]
        out = {}
        for k in ("roc_auc", "pr_auc", "brier", "f1", "recall", "precision"):
            vals = np.array([r[f"{k}{suffix}"] for r in sub], dtype=float)
            out[k] = (float(vals.mean()), float(vals.std()))
        return out

    md_lines = [
        "# P0.2 Label sensitivity — nhãn A (có thuốc) vs B (không thuốc)",
        "",
        f"- Dataset: `{args.data}`",
        f"- Positive rate: A = {overlap['positive_rate_A']:.1%}, "
        f"B = {overlap['positive_rate_B']:.1%}",
        f"- Nhãn đổi dấu: {overlap['changed_total']}/{overlap['n']} ca "
        f"(chỉ dương ở A nhờ thuốc: {overlap['positives_only_in_A']})",
        "",
        "| Model | ROC-AUC A | ROC-AUC B | Δ AUC | PR-AUC A | PR-AUC B | Δ PR | Brier A | Brier B | Δ Brier | F1 A | F1 B | Δ F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    agg_out = {}
    for key in args.models:
        a, b = agg(key, "_A"), agg(key, "_B")
        agg_out[key] = {"A": a, "B": b}
        da = {k: b[k][0] - a[k][0] for k in a}
        db = {k: b[k][1] + a[k][1] for k in a}
        fmt = lambda t: f"{t[0]:.4f}±{t[1]:.4f}"  # noqa: E731
        dms = lambda k: f"{da[k]:+.4f}±{db[k]:.4f}"  # noqa: E731
        md_lines.append(
            f"| {key} | {fmt(a['roc_auc'])} | {fmt(b['roc_auc'])} | {dms('roc_auc')} "
            f"| {fmt(a['pr_auc'])} | {fmt(b['pr_auc'])} | {dms('pr_auc')} "
            f"| {fmt(a['brier'])} | {fmt(b['brier'])} | {dms('brier')} "
            f"| {fmt(a['f1'])} | {fmt(b['f1'])} | {dms('f1')} |"
        )
    md_lines += [
        "",
        "> Cùng split, cùng seed; nhãn B bỏ thành phần \"đang dùng thuốc HA\".",
        "> Dù A ~ B hay A != B, giới hạn D1 (nhãn định nghĩa bởi chính feature)",
        "> vẫn giữ nguyên: model tái lập phân tầng lâm sàng, chưa dự báo biến cố.",
    ]
    summary["aggregate"] = agg_out
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nKết quả lưu tại: {OUT_DIR}/")


if __name__ == "__main__":
    main()
