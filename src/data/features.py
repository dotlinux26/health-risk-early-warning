"""Đặc trưng hóa chuỗi thời gian cho mô hình Tầng 3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


def rolling_features(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: list[int] = (7, 30, 90),
) -> pd.DataFrame:
    """Sinh rolling mean / std / delta / %change cho từng chỉ số.

    Rolling chỉ dùng dữ liệu quá khứ nên không rò rỉ dữ liệu tương lai.
    """
    df = df.copy().set_index("timestamp")
    for col in value_cols:
        s = df[col]
        for w in windows:
            df[f"{col}_mean_{w}d"] = s.rolling(w, min_periods=2).mean()
            df[f"{col}_std_{w}d"] = s.rolling(w, min_periods=2).std()
        df[f"{col}_delta_1"] = s.diff()
        df[f"{col}_pct_1"] = s.pct_change()
    return df.reset_index()


def slope_features(
    df: pd.DataFrame,
    value_cols: list[str],
    window: int = 30,
) -> pd.DataFrame:
    """Độ dốc xu hướng (linear regression slope) trong cửa sổ gần nhất."""
    df = df.copy()
    for col in value_cols:
        slope = df[col].rolling(window, min_periods=5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
        )
        df[f"{col}_slope_{window}d"] = slope
    return df


def ewma_trend(df: pd.DataFrame, value_cols: list[str], alpha: float = 0.2) -> pd.DataFrame:
    """Trung bình động lũy thừa (EWMA) — bắt xu hướng trượt nhẹ, ít nhiễu."""
    df = df.copy().set_index("timestamp")
    for col in value_cols:
        df[f"{col}_ewma"] = df[col].ewm(alpha=alpha, min_periods=5).mean()
    return df.reset_index()


def compute_zscore(df: pd.DataFrame, value_cols: list[str], config: Config = Config()) -> pd.DataFrame:
    """Z-Score cá nhân hóa: Z = (X - μ_base) / σ_base."""
    df = df.copy()
    for col in value_cols:
        mean_col, std_col = f"{col}_base_mean", f"{col}_base_std"
        if mean_col in df.columns and std_col in df.columns:
            std_safe = df[std_col].replace(0, np.nan)
            df[f"{col}_zscore"] = (df[col] - df[mean_col]) / std_safe
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    config: Config = Config(),
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Pipeline đặc trưng hóa trọn gói cho một bệnh nhân (wide format)."""
    numeric = value_cols or [
        c for c in df.columns if c not in {"timestamp", "patient_id"}
        and not c.endswith(("_base_mean", "_base_std", "_zscore"))
        and not c.endswith("_missing_flag")
    ]
    df = compute_zscore(df, numeric, config)
    df = rolling_features(df, numeric)
    df = slope_features(df, numeric)
    df = ewma_trend(df, numeric)
    return df
