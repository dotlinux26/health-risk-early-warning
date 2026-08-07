"""Isolation Forest đa chiều — phát hiện thay đổi ĐỒNG THỜI nhiều chỉ số."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import Config
from src.tier1_anomaly import AnomalyRecord


def detect_isolation_forest(
    df: pd.DataFrame,
    value_cols: list[str],
    config: Config = Config(),
) -> list[AnomalyRecord]:
    """Chạy Isolation Forest trên cửa sổ gần nhất của các chỉ số.

    Trả về AnomalyRecord cho từng chỉ số tại điểm cuối nếu điểm đó bị cô lập.
    """
    if len(df) < 10:
        return []

    window = df.tail(config.isolation_forest_metric_window)
    X = window[value_cols].dropna()
    if len(X) < 10:
        return []

    model = IsolationForest(
        contamination=config.isolation_forest_contamination,
        random_state=config.random_state,
    )
    pred = model.fit_predict(X)  # -1 = bất thường
    if pred[-1] != -1:
        return []

    records: list[AnomalyRecord] = []
    last = X.iloc[-1]
    for col in value_cols:
        mu = float(window[col].mean())
        records.append(
            AnomalyRecord(
                metric=col,
                current=float(last[col]),
                baseline_mean=round(mu, 2),
                z_score=None,
                window_days=config.isolation_forest_metric_window,
                direction="up" if float(last[col]) > mu else "down",
                trend="unknown",
                flagged=True,
            )
        )
    return records
