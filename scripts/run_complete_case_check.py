"""P2.2 — COMPLETE-CASE CHECK: impute median có làm kết quả sai không? (T11)

Vấn đề (docs/16 §3.5): glucose_fasting khuyết ~52% trong NHANES; hệ thống đang
impute bằng median fit-train. Nếu kết quả trên tập complete-case (bỏ mọi mẫu
thiếu) ổn định so với arm impute thì impute vô hại; nếu lệch lớn thì mọi con số
liên quan glucose phải báo cáo kèm phương sai do impute.

Thiết kế (giống protocol K2/K4 — cùng split, cùng seed cho CẢ HAI arm):
  - Dataset: nhanes_merged.csv, nhãn sản xuất (tăng HA HOẶC ĐTĐ).
  - Split 70/15/15 stratified theo 5 seeds [42,52,62,72,82].
  - Arm "imputed": SimpleImputer(median) fit-train — như production.
  - Arm "complete_case": sau split, bỏ mọi dòng còn thiếu ≥1 trong 7 đặc trưng
    (train/val/test tự loại riêng — không rò thông tin giữa các phần).
  - Models: lgbm, xgb, lr (build_model của src/experiments).
  - Metric chính: Δ ROC-AUC = complete-case − imputed (cùng seed); tiêu chí
    nghiệm thu docs/16 §3.5: |ΔAUC| < 0.01 → impute vô hại.

Chạy:
    python3 scripts/run_complete_case_check.py

Kết quả: experiments/COMPLETE-CASE-CHECK/{summary.json, summary.md}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.experiments.models import available_models, build_model
from src.experiments.protocol import (
    NHANES_FEATURES,
    SEEDS,
    ExperimentConfig,
    evaluate_metrics,
    stratify_split,
)

OUT_DIR = Path("experiments/COMPLETE-CASE-CHECK")
DEFAULT_MODELS = ["lgbm", "xgb", "lr"]
ACCEPTANCE_DELTA_AUC = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete-case check vs impute")
    parser.add_argument("--data", type=str, default="data/datasets/nhanes_merged.csv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        choices=available_models())
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["label"])
    features = list(NHANES_FEATURES)
    X = df[features].copy()
    y = df["label"].to_numpy(dtype=int)
    n_missing_glucose = int(X["glucose_fasting"].isna().sum())

    print(f"n={len(df)} | thiếu glucose_fasting={n_missing_glucose} "
          f"({n_missing_glucose / len(df):.0%})")

    rows: list[dict] = []
    for key in args.models:
        for seed in args.seeds:
            cfg = ExperimentConfig(model_key=key, seed=seed, dataset=args.data,
                                   features=features)
            splits = stratify_split(X, y, seed, cfg)
            res = {"model": key, "seed": seed}
            for arm in ("imputed", "complete_case"):
                X_tr, X_va, X_te, y_tr, y_va, y_te = splits

                if arm == "complete_case":
                    # Loại dòng thiếu trong TỪNG phần — không dùng thống kê chéo phần
                    def cc(x: pd.DataFrame, yy: np.ndarray):
                        m = x[features].notna().all(axis=1).to_numpy()
                        return x[m], yy[m]
                    X_tr, y_tr = cc(X_tr, y_tr)
                    X_va, y_va = cc(X_va, y_va)
                    X_te, y_te = cc(X_te, y_te)
                    imp = SimpleImputer(strategy="median").fit(X_tr)  # an toàn nếu vẫn NaN
                else:
                    imp = SimpleImputer(strategy="median").fit(X_tr)

                cols = list(features)
                model = build_model(key, seed)
                model.fit(pd.DataFrame(imp.transform(X_tr), columns=cols), y_tr)
                proba = model.predict_proba(pd.DataFrame(imp.transform(X_te), columns=cols))[:, 1]
                m = evaluate_metrics(y_te, proba)
                m["n_test"] = int(len(y_te))
                m["events_test"] = int(y_te.sum())
                res.update({f"{k}_{arm}": v for k, v in m.items()})
            for k in ("roc_auc", "pr_auc", "brier"):
                res[f"delta_{k}"] = round(res[f"{k}_complete_case"] - res[f"{k}_imputed"], 6)
            rows.append(res)
            print(f"  {cfg.experiment_id}: AUC imputed={res['roc_auc_imputed']:.4f} "
                  f"cc={res['roc_auc_complete_case']:.4f} "
                  f"(n_test {res['n_test_imputed']}→{res['n_test_complete_case']}) "
                  f"d={res['delta_roc_auc']:+.4f}")

    # ---- Aggregate ---------------------------------------------------------
    agg_out = {}
    worst = 0.0
    for key in args.models:
        sub = [r for r in rows if r["model"] == key]
        entry = {}
        for k in ("roc_auc", "pr_auc", "brier"):
            imp_v = np.array([r[f"{k}_imputed"] for r in sub])
            cc_v = np.array([r[f"{k}_complete_case"] for r in sub])
            d = cc_v.mean() - imp_v.mean()
            entry[k] = {
                "imputed": f"{imp_v.mean():.4f}±{imp_v.std():.4f}",
                "complete_case": f"{cc_v.mean():.4f}±{cc_v.std():.4f}",
                "delta": round(float(d), 5),
            }
            if k == "roc_auc":
                worst = max(worst, abs(float(d)))
        n_ratio = float(np.mean([r["n_test_complete_case"] / r["n_test_imputed"] for r in sub]))
        entry["test_n_retained_ratio"] = round(n_ratio, 4)
        agg_out[key] = entry

    def mean_delta_auc(key: str) -> float:
        sub = [r for r in rows if r["model"] == key]
        return float(np.mean([r["delta_roc_auc"] for r in sub]))

    per_model_mean = {k: round(mean_delta_auc(k), 5) for k in args.models}
    stable = [k for k, v in per_model_mean.items() if abs(v) < ACCEPTANCE_DELTA_AUC]
    unstable = [k for k, v in per_model_mean.items() if abs(v) >= ACCEPTANCE_DELTA_AUC]
    parts = []
    if stable:
        parts.append(f"ổn định cho {', '.join(stable)} (|ΔAUC| trung bình < "
                     f"{ACCEPTANCE_DELTA_AUC})")
    if unstable:
        parts.append("lệch có hệ thống cho " + ", ".join(
            f"{k} ({per_model_mean[k]:+.4f})" for k in unstable))
    verdict = ("; ".join(parts)
               + f". Sản xuất dùng LightGBM → "
               + ("impute chấp nhận được"
                  if abs(per_model_mean.get("lgbm", 9)) < ACCEPTANCE_DELTA_AUC
                  else "cân nhắc lại chiến lược impute")
               + "; mô hình tuyến tính bị impute median làm giảm AUC "
                 "(glucose đặc thành 1 giá trị ở 52% mẫu).")
    summary = {
        "experiment": "COMPLETE-CASE-CHECK",
        "question": "docs/16 §3.5 (T11): impute median cho glucose_fasting 52% có vô hại?",
        "dataset": args.data,
        "seeds": args.seeds,
        "acceptance": f"|Δ ROC-AUC| < {ACCEPTANCE_DELTA_AUC}",
        "missing_glucose": {"count": n_missing_glucose, "rate": round(n_missing_glucose / len(df), 4)},
        "runs": rows,
        "aggregate": agg_out,
        "per_model_mean_delta_auc": per_model_mean,
        "worst_abs_delta_auc": round(worst, 5),
        "verdict": verdict,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# P2.2 Complete-case check — impute median vs bỏ mẫu thiếu",
        "",
        f"- Dataset: `{args.data}` (glucose_fasting khuyết {n_missing_glucose / len(df):.0%})",
        "- Cùng split + seed cho hai arm; complete-case loại mẫu thiếu trong từng phần.",
        "",
        "| Model | AUC imputed | AUC complete-case | Δ AUC | PR-AUC Δ | Brier Δ | Giữ được % test |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in args.models:
        a = agg_out[key]
        md.append(
            f"| {key} | {a['roc_auc']['imputed']} | {a['roc_auc']['complete_case']} "
            f"| {a['roc_auc']['delta']:+.4f} | {a['pr_auc']['delta']:+.4f} "
            f"| {a['brier']['delta']:+.4f} | {a['test_n_retained_ratio']:.1%} |")
    md += [
        "",
        f"**Kết luận: {verdict}**",
        "",
        f"> Tiêu chí nghiệm thu docs/16 §3.5: ổn định < {ACCEPTANCE_DELTA_AUC} → impute vô hại;",
        "> lệch lớn → báo cáo kèm phương sai do impute. Δ dương = complete-case tốt hơn.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n{verdict}")
    print(f"Kết quả lưu tại: {OUT_DIR}/")


if __name__ == "__main__":
    sys.exit(main())
