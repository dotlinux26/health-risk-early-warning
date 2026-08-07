"""Mô hình điểm rủi ro: LightGBM (chính) và LSTM (đối chứng).

Thiết kế: nhận ma trận đặc trưng chuỗi thời gian (đã sinh ở data/features.py),
học nhãn nhị phân "có bệnh trong k cửa sổ tới" hoặc điểm nguy cơ liên tục.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

try:
    from lightgbm import LGBMClassifier

    _HAS_LGBM = True
except ImportError:  # pragma: no cover - fallback
    _HAS_LGBM = False

from src.config import Config


class RiskModel:
    """Wrapper nhất quán cho mô hình điểm rủi ro (LightGBM mặc định)."""

    def __init__(self, config: Config = Config(), name: str | None = None) -> None:
        self.config = config
        self.name = name or config.model_name
        if not _HAS_LGBM:
            raise RuntimeError("Chưa cài lightgbm. Chạy: pip install lightgbm")
        self.model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=16,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.random_state,
            verbosity=-1,
        )

    def train(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Huấn luyện trên ma trận đặc trưng."""
        self.model.fit(X, y)

    def predict_proba_score(self, X: pd.DataFrame) -> np.ndarray:
        """Trả xác suất (điểm) rủi ro trong [0, 1]."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict[str, float]:
        y_pred = self.predict_proba_score(X)
        return {
            "auc": float(roc_auc_score(y, y_pred)),
            "mean_risk": float(y_pred.mean()),
        }

    def save(self, path: Path) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path, config: Config = Config()) -> "RiskModel":
        obj = cls(config)
        obj.model = joblib.load(path)
        return obj
