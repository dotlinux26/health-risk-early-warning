"""Phát hiện xu hướng dài hạn bằng EWMA và phân rã chuỗi thời gian (STL)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_ewma_crossing(df: pd.DataFrame, value_cols: list[str], alpha: float = 0.2) -> dict[str, str]:
    """So sánh EWMA mới nhất với trung bình cửa sổ dài → xác định xu hướng.

    Trả về {metric: "rising" | "falling" | "stable"}.
    """
    result: dict[str, str] = {}
    for col in value_cols:
        s = df[col].dropna()
        if len(s) < 10:
            result[col] = "stable"
            continue
        ewma = s.ewm(alpha=alpha, min_periods=5).mean()
        long_mean = s.rolling(len(s) // 2, min_periods=5).mean()
        if np.isnan(ewma.iloc[-1]) or np.isnan(long_mean.iloc[-1]):
            result[col] = "stable"
            continue
        delta = (ewma.iloc[-1] - long_mean.iloc[-1]) / (abs(long_mean.iloc[-1]) + 1e-9)
        if delta > 0.02:
            result[col] = "rising"
        elif delta < -0.02:
            result[col] = "falling"
        else:
            result[col] = "stable"
    return result


def stl_decomposition(df: pd.DataFrame, value_cols: list[str], period: int = 7) -> dict[str, pd.Series]:
    """Phân rã chuỗi bằng STL (Seasonal-Trend using LOESS).

    Yêu cầu statsmodels. Nếu không có thư viện, trả về dict rỗng và
    hệ thống vẫn hoạt động nhờ EWMA (giảm phụ thuộc).
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        return {}

    result: dict[str, pd.Series] = {}
    for col in value_cols:
        s = df[col].dropna()
        if len(s) < period * 2:
            continue
        try:
            decomp = STL(s, period=period, robust=True).fit()
            result[col] = decomp.trend
        except Exception:
            continue
    return result
