"""Huấn luyện mô hình LightGBM trên dữ liệu tổng hợp có nhãn.

Chạy:
    python -m src.tier3_risk.ml.train --n-per-condition 120 --seed 42
    python -m src.tier3_risk.ml.train --n-per-condition 120 --seed 42 --out data/models/risk_lgbm

Sinh bệnh nhân (4 trạng thái) -> ma trận đặc trưng 60 ngày -> LightGBM nhị phân
-> lưu model + tên cột đặc trưng + báo cáo AUC/AUPRC trên test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.tier3_risk.ml.risk_model import RiskModel
from src.tier3_risk.ml.synthetic_data import (
    CONDITIONS,
    STATE_LABEL,
    build_train_matrix,
    generate_labeled_dataset,
)

FEATURE_COLS = list(CONDITIONS["healthy"].metrics.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện LightGBM cảnh báo sớm")
    parser.add_argument("--n-per-condition", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-window", type=int, default=60)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--out", type=str, default="data/models/risk_lgbm")
    args = parser.parse_args()

    cfg = Config()
    rng = np.random.default_rng(args.seed)

    print("1) Sinh dữ liệu tổng hợp có nhãn ...")
    wide_dfs, y, states = generate_labeled_dataset(
        n_per_condition=args.n_per_condition,
        feature_window=args.feature_window,
        seed=args.seed,
    )
    y = np.asarray(y, dtype=int)
    states = np.asarray(states)
    n = len(wide_dfs)
    pos = int(y.sum())
    print(f"   {n} bệnh nhân | dương tính (sự kiện 61-{args.feature_window + 30} ngày): {pos} ({pos / n:.0%})")
    print(f"   Phân bố trạng thái: { {s: int((states == s).sum()) for s in CONDITIONS} }")

    print("2) Đặc trưng hóa (window đầu) ...")
    X = build_train_matrix(wide_dfs, feature_window=args.feature_window, value_cols=FEATURE_COLS, config=cfg)

    perm = rng.permutation(n)
    n_test = int(n * args.test_ratio)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    print(f"   Train {len(train_idx)} | Test {len(test_idx)} | dương tính test {y_test.sum()}")

    print("3) Huấn luyện LightGBM ...")
    model = RiskModel(cfg)
    model.train(X_train, y_train)

    from sklearn.metrics import average_precision_score, roc_auc_score

    y_pred = model.predict_proba_score(X_test)
    auc = float(roc_auc_score(y_test, y_pred))
    auprc = float(average_precision_score(y_test, y_pred))
    print(f"   AUC = {auc:.4f} | AUPRC = {auprc:.4f}")

    # Phân bố điểm theo trạng thái thật (kiểm tra mô hình không học nhiễu)
    pred_df = pd.DataFrame(
        {"y_true": y_test, "score": y_pred, "state": states[test_idx]}
    )
    print("   Điểm trung bình theo trạng thái thật (test):")
    for s, row in pred_df.groupby("state")["score"].mean().sort_values(ascending=False).items():
        print(f"     {STATE_LABEL[s]:<20} {row:.3f}")

    print("4) Lưu model + feature names ...")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.with_suffix(".joblib"))
    meta = {
        "feature_names": list(X.columns),
        "feature_window_days": args.feature_window,
        "n_patients": n,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "seed": args.seed,
        "auc_test": round(auc, 4),
        "auprc_test": round(auprc, 4),
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    (out.parent / f"{out.name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"   Model: {out}.joblib")
    print(f"   Meta : {out}_meta.json")
    print(f"   Đặc trưng: {len(X.columns)} cột")

    # Đặc trưng quan trọng nhất (gini của LightGBM)
    imp = pd.Series(model.model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("   Top 15 đặc trưng theo importance:")
    for k, v in imp.head(15).items():
        print(f"     {k:<36} {v}")


if __name__ == "__main__":
    main()
