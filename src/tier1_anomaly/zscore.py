"""Z-Score cá nhân hóa (so với đường cơ sở của chính cá nhân, không phải quần thể)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.tier1_anomaly import AnomalyRecord


def detect_zscore(
    df: pd.DataFrame,
    value_cols: list[str],
    config: Config = Config(),
) -> list[AnomalyRecord]:
    """Đánh giá giá trị mới nhất so với μ/σ cá nhân trong cửa sổ.

    df kỳ vọng đã có cột {metric}_base_mean, {metric}_base_std
    (tạo từ build_baseline trong src/data/preprocess.py).
    """
    records: list[AnomalyRecord] = []
    if df.empty:
        return records

    last = df.iloc[-1]
    for col in value_cols:
        mean_col, std_col = f"{col}_base_mean", f"{col}_base_std"
        if mean_col not in df.columns or std_col not in df.columns:
            continue
        current = float(last[col])
        if pd.isna(current):
            continue
        mu = float(last[mean_col])
        sigma = float(last[std_col])
        if np.isnan(mu) or np.isnan(sigma) or sigma == 0:
            continue

        z = (current - mu) / sigma
        flagged = abs(z) >= config.zscore_threshold
        direction = "up" if z > 0 else "down"

        # Xu hướng: so sánh trung bình nửa trước/nửa sau cửa sổ
        half = len(df) // 2 or 1
        trend = "stable"
        if len(df) >= 6:
            first_half = df[col].iloc[:half].mean()
            second_half = df[col].iloc[half:].mean()
            if second_half > first_half * 1.02:
                trend = "rising"
            elif second_half < first_half * 0.98:
                trend = "falling"

        records.append(
            AnomalyRecord(
                metric=col,
                current=current,
                baseline_mean=mu,
                z_score=round(z, 2),
                window_days=config.zscore_window_days,
                direction=direction,
                trend=trend,
                flagged=flagged,
            )
        )
    return records
