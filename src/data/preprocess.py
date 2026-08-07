"""Làm sạch và căn chỉnh chuỗi thời gian."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


def remove_outlier_rows(df: pd.DataFrame, z_cutoff: float = 8.0) -> pd.DataFrame:
    """Loại bỏ giá trị thô bất thường (lỗi nhập liệu) — theo chỉ số trong CHÍNH chuỗi.

    Dùng ngưỡng rất cao (mặc định 8σ) để chỉ bỏ lỗi nhập, KHÔNG xóa biến động sinh lý.
    """
    clean = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        mu, sigma = df[col].mean(), df[col].std()
        if sigma == 0 or np.isnan(sigma):
            continue
        clean[col] = df[col].mask(((df[col] - mu).abs() > z_cutoff * sigma))
    return clean


def resample_to_daily(df: pd.DataFrame, date_col: str = "timestamp") -> pd.DataFrame:
    """Căn chỉnh chuỗi về mốc ngày (lấy giá trị cuối cùng trong ngày)."""
    df = df.set_index(date_col).resample("D").last().reset_index()
    return df


def impute_missing(
    df: pd.DataFrame,
    config: Config = Config(),
    value_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Nội suy giá trị thiếu theo thời gian.

    - Nếu tỷ lệ thiếu > missing_ratio_limit trong cửa sổ -> giữ NaN và đánh dấu
      (hệ thống sẽ báo INSUFFICIENT_DATA thay vì đoán bừa).
    - Ngược lại: nội suy tuyến tính theo thời gian (time-aware) + backfill/LOCF.
    """
    df = df.copy()
    cols = value_cols or [c for c in df.columns if c != "timestamp"]
    for col in cols:
        ratio = df[col].isna().mean()
        if ratio > config.missing_ratio_limit:
            # đánh dấu thiếu nặng -> để pipeline xử lý an toàn
            df[f"{col}_missing_flag"] = df[col].isna()
        else:
            ts = df["timestamp"]
            known = ~df[col].isna()
            if known.sum() >= 2:
                df[col] = np.interp(
                    ts.astype("int64"),
                    ts[known].astype("int64"),
                    df.loc[known, col],
                )
            df[col] = df[col].ffill().bfill()
    return df


def build_baseline(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """Tính đường cơ sở cá nhân (μ, σ) bằng cửa sổ trượt TRÁI — không nhìn tương lai.

    Sinh ra các cột {metric}_base_mean, {metric}_base_std dùng cho Z-Score.
    """
    df = df.copy()
    numeric = [c for c in df.columns if c not in {"timestamp"}]
    for col in numeric:
        base = df.set_index("timestamp")[col]
        rolling_mean = base.rolling(f"{window_days}D", min_periods=5).mean()
        rolling_std = base.rolling(f"{window_days}D", min_periods=5).std()
        df[f"{col}_base_mean"] = rolling_mean.to_numpy()
        df[f"{col}_base_std"] = rolling_std.to_numpy()
    return df
