"""Chạy benchmark và ghi evidence package cho từng model/seed.

Output layout:
    experiments/
    ├── EXP-ML-<MODEL>-<SEED>/
    │   ├── config.json
    │   ├── dataset_manifest.json
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── roc_curve.png / pr_curve.png / calibration_curve.png
    │   ├── feature_importance.csv
    │   └── model.joblib
    ├── summary.json / summary.md / summary.csv
    └── README.md (hướng dẫn sử dụng)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score

from src.experiments.models import build_model
from src.experiments.protocol import (
    NHANES_FEATURES,
    ExperimentConfig,
    aggregate_seeds,
    evaluate_metrics,
    stratify_split,
)

EXPERIMENTS_DIR = Path("experiments")


def _fmt_metric_table(agg: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | Specificity | F1 | Brier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in agg.items():
        cells = []
        for k in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "specificity", "f1", "brier"):
            v = m[k]
            cells.append(f"{v['mean']:.3f}±{v['std']:.3f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


class BenchmarkRunner:
    """Chạy một model trên nhiều seed, ghi evidence, trả về dòng kết quả."""

    def __init__(self, config: ExperimentConfig, out_dir: Path = EXPERIMENTS_DIR) -> None:
        self.config = config
        self.out_dir = Path(out_dir)
        self.exp_dir = self.out_dir / config.experiment_id

    def run(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        seed = self.config.seed
        X_tr, X_va, X_te, y_tr, y_va, y_te = stratify_split(X, y, seed, self.config)

        # Preprocessing riêng cho từng model (kháng NaN): imputer fit trên TRAIN
        # để tránh data leakage. Model nào tự xử lý NaN (LightGBM) vẫn được
        # impute — nhất quán về information boundary, không đổi thứ tự chia.
        imp = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X.columns)
        X_va = pd.DataFrame(imp.transform(X_va), columns=X.columns)
        X_te = pd.DataFrame(imp.transform(X_te), columns=X.columns)

        model = build_model(self.config.model_key, seed)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_seconds = round(time.time() - t0, 2)

        proba_te = model.predict_proba(X_te)[:, 1]
        proba_va = model.predict_proba(X_va)[:, 1]

        metrics = evaluate_metrics(y_te, proba_te)
        metrics["train_seconds"] = train_seconds
        metrics["n_train"] = int(len(X_tr))
        metrics["n_val"] = int(len(X_va))
        metrics["n_test"] = int(len(X_te))

        # Evidence package
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "config.json").write_text(
            json.dumps(self.config.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (self.exp_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pred_df = pd.DataFrame(
            {"y_true": y_te, "proba": proba_te, "seed": seed}
        )
        pred_df.to_csv(self.exp_dir / "predictions.csv", index=False)
        joblib.dump(model, self.exp_dir / "model.joblib")

        self._save_curves(y_te, proba_te)
        self._save_importance(model, X_tr, y_tr, X_va, y_va)

        return metrics

    def _save_curves(self, y_true: np.ndarray, proba: np.ndarray) -> None:
        fpr, tpr, _ = roc_curve(y_true, proba)
        precision, recall, _ = precision_recall_curve(y_true, proba)
        prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(fpr, tpr, label="ROC")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set_title("ROC Curve")
        axes[0].set_xlabel("FPR")
        axes[0].set_ylabel("TPR")
        axes[1].plot(recall, precision, label="PR")
        axes[1].set_title("PR Curve")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[2].plot(prob_pred, prob_true, marker="o")
        axes[2].plot([0, 1], [0, 1], "--", color="gray")
        axes[2].set_title("Calibration")
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(self.exp_dir / "curves.png", dpi=120)
        plt.close(fig)

    def _save_importance(
        self,
        model,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        X_va: pd.DataFrame,
        y_va: np.ndarray,
    ) -> None:
        """Feature importance theo loại model; fallback permutation importance."""
        importances: np.ndarray | None = None
        feats = list(X_tr.columns)

        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_)[0])
        elif hasattr(model, "net_"):
            importances = self._permutation_importance(model, X_va, y_va)

        if importances is None:
            importances = self._permutation_importance(model, X_va, y_va)

        df = pd.DataFrame({"feature": feats, "importance": importances})
        df = df.sort_values("importance", ascending=False)
        df.to_csv(self.exp_dir / "feature_importance.csv", index=False)

    def _permutation_importance(
        self, model, X_va: pd.DataFrame, y_va: np.ndarray
    ) -> np.ndarray:
        base = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
        out = []
        for col in X_va.columns:
            X_p = X_va.copy()
            X_p[col] = X_p[col].median()
            p = roc_auc_score(y_va, model.predict_proba(X_p)[:, 1])
            out.append(max(0.0, base - p))
        return np.asarray(out, dtype=float)


def run_benchmark(
    dataset_path: str | Path,
    model_keys: list[str] | None = None,
    seeds: list[int] | None = None,
    out_dir: Path = EXPERIMENTS_DIR,
) -> dict:
    """Chạy benchmark cho các model (mặc định: toàn bộ model khả dụng)."""
    from src.experiments.models import available_models

    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["label"])
    features = list(NHANES_FEATURES)
    X = df[features].copy()
    y = df["label"].to_numpy(dtype=int)

    keys = model_keys or available_models()
    seeds = seeds or [42]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: dict[str, list[dict]] = {}
    per_seed: dict[str, list[dict]] = {}
    for key in keys:
        per_seed[key] = []
        for seed in seeds:
            cfg = ExperimentConfig(
                model_key=key,
                seed=seed,
                dataset=str(dataset_path),
                features=features,
            )
            metrics = BenchmarkRunner(cfg, out_dir).run(X, y)
            metrics["seed"] = seed
            per_seed[key].append(metrics)
            print(f"  {cfg.experiment_id}: AUC={metrics['roc_auc']:.4f}")
        rows_by_model[key] = per_seed[key]

    agg = {key: aggregate_seeds(rows) for key, rows in rows_by_model.items()}

    # --- Mức độ đầy đủ dữ liệu: đánh dấu rõ chỉ số nào bị khuyết nhiều -------
    # Benchmark phải minh bạch về information boundary: một chỉ số missing >20%
    # làm suy yếu độ tin cậy của mô hình trên chỉ số đó, kết quả AUC cao có thể
    # đến từ các chỉ số còn lại. Ghi thật vào summary để người đọc tự đánh giá.
    missing_rates = {f: float(round(X[f].isna().mean(), 4)) for f in features}
    data_completeness = {
        "missing_rates": missing_rates,
        "flags": {
            f: (f"MISSING_CAO ({missing_rates[f]:.0%})" if missing_rates[f] > 0.20 else
                f"thiếu_1_phần ({missing_rates[f]:.0%})" if missing_rates[f] > 0.05 else
                "đầy_đủ")
            for f in features
        },
    }
    # Cảnh báo độ tin cậy tổng thể dựa trên mức missing trung bình
    avg_missing = float(np.mean(list(missing_rates.values())))
    if avg_missing > 0.20:
        data_completeness["confidence"] = "THAP — nhiều chỉ số bị khuyết, kết quả cần đọc thận trọng"
    elif avg_missing > 0.05:
        data_completeness["confidence"] = "TRUNG_BINH — một phần chỉ số bị khuyết, đã impute median"
    else:
        data_completeness["confidence"] = "CAO — dữ liệu gần như đầy đủ"

    summary = {
        "dataset": str(dataset_path),
        "n": int(len(df)),
        "positive": int(y.sum()),
        "features": features,
        "seeds": seeds,
        "data_completeness": data_completeness,
        "models": agg,
        "per_seed": {k: [{"seed": r["seed"], "roc_auc": r["roc_auc"]} for r in rows] for k, rows in rows_by_model.items()},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md = "# Bảng tổng hợp benchmark\n\n" + _fmt_metric_table(agg) + "\n"
    md += "\n## Mức độ đầy đủ dữ liệu\n"
    md += f"- **Confidence tổng thể: {data_completeness['confidence']}**\n"
    for f in features:
        md += f"- `{f}`: {data_completeness['flags'][f]}\n"
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    summary_rows = []
    for key, rows in rows_by_model.items():
        for r in rows:
            r["model"] = key
            summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    return summary