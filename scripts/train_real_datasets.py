"""Train & đánh giá LightGBM trên DATASET THẬT (Pima diabetes, Cleveland heart).

Chạy:
    python scripts/train_real_datasets.py

So với train trên dữ liệu tổng hợp (src/tier3_risk/ml/train.py), script này dùng dữ
liệu thực thu thập lâm sàng (UCI), cross-validation 5-folds theo bệnh nhân để
đánh giá khách quan hơn AUC/AUPRC. Kết quả ghi ra report/train_real_results.json
phục vụ báo cáo docs/06.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer

from src.tier3_risk.ml.risk_model import RiskModel
from src.config import Config


def load_pima(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    X = df.drop(columns=["outcome"])
    y = df["outcome"].to_numpy(dtype=int)
    # 0 trong các cột sinh hóa là "không đo được" -> coi như khuyết
    for col in ["glucose", "bp", "skin_thickness", "insulin", "bmi"]:
        X.loc[X[col] == 0, col] = np.nan
    return X, y, list(X.columns)


def load_cleveland(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    for col in ["ca", "thal"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    y = (df["num"] > 0).to_numpy(dtype=int)
    X = df.drop(columns=["num"])
    return X, y, list(X.columns)


def evaluate_cv(model_factory, X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> dict:
    """Stratified 5-fold CV -> AUC/AUPRC trung bình + per-fold."""
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, auprcs = [], []
    fold_rows = []
    for fold, (tr, te) in enumerate(skf.split(X_imp, y)):
        model = model_factory()
        model.train(pd.DataFrame(X_imp[tr], columns=X.columns), y[tr])
        proba = model.predict_proba_score(pd.DataFrame(X_imp[te], columns=X.columns))
        auc = float(roc_auc_score(y[te], proba))
        auprc = float(average_precision_score(y[te], proba))
        aucs.append(auc)
        auprcs.append(auprc)
        fold_rows.append({"fold": fold + 1, "auc": round(auc, 4), "auprc": round(auprc, 4)})
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "folds": fold_rows,
    }


def top_features(model, X: pd.DataFrame) -> list[dict]:
    imp = pd.Series(model.model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return [{"feature": k, "importance": int(v)} for k, v in imp.head(10).items()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM trên dataset thật")
    parser.add_argument("--data", type=str, default="data/datasets")
    parser.add_argument("--out", type=str, default="report/train_real_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    cfg = Config()

    def factory() -> RiskModel:
        return RiskModel(cfg)

    results: dict[str, dict] = {"config": {"seed": args.seed, "framework": "lightgbm"}}

    print("== Pima Indians Diabetes (UCI) ==")
    X, y, cols = load_pima(data_dir / "pima_diabetes.csv")
    r = evaluate_cv(factory, X, y, args.seed)
    print(f"   n={len(y)} | positive={y.sum()} | AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f} | AUPRC={r['auprc_mean']:.4f}")
    model = factory()
    model.train(pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns), y)
    results["pima_diabetes"] = {"n": int(len(y)), "positive": int(y.sum()),
                                "cv": r, "top_features": top_features(model, X)}

    print("== Cleveland Heart Disease (UCI) ==")
    X, y, cols = load_cleveland(data_dir / "heart_cleveland.csv")
    r = evaluate_cv(factory, X, y, args.seed)
    print(f"   n={len(y)} | positive={y.sum()} | AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f} | AUPRC={r['auprc_mean']:.4f}")
    model = factory()
    model.train(pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns), y)
    results["heart_cleveland"] = {"n": int(len(y)), "positive": int(y.sum()),
                                  "cv": r, "top_features": top_features(model, X)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kết quả -> {out}")


if __name__ == "__main__":
    main()
