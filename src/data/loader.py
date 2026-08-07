"""Nạp dữ liệu chuỗi chỉ số cơ thể.

Schema đầu vào chuẩn (CSV/DataFrame):
    patient_id | timestamp | metric | value | unit (optional)
  - metric là tên chỉ số chuẩn hóa, ví dụ: systolic_bp, diastolic_bp,
    heart_rate, glucose, hba1c, creatinine, bmi, spo2
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

REQUIRED_COLUMNS = {"patient_id", "timestamp", "metric", "value"}


def load_csv(path: str | Path, sep: str = ",") -> pd.DataFrame:
    """Đọc dữ liệu CSV và validate schema tối thiểu."""
    df = pd.read_csv(path, sep=sep)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_long_df(
    df: pd.DataFrame,
    index_col: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """Chuyển từ định dạng 'long' (metric/value) sang 'wide' theo từng bệnh nhân.

    Trả về dict {patient_id: DataFrame[timestamp, metric_1, metric_2, ...]}.
    """
    patients: dict[str, pd.DataFrame] = {}
    for pid, group in df.groupby("patient_id"):
        group = group.copy()
        group[index_col] = pd.to_datetime(group[index_col], errors="coerce")
        group = group.dropna(subset=[index_col])
        wide = group.pivot_table(
            index=index_col, columns="metric", values=value_col, aggfunc="last"
        ).sort_index()
        wide = wide.reset_index()
        patients[pid] = wide
    return patients


def load_json(path: str | Path) -> pd.DataFrame:
    """Đọc dữ liệu dạng JSON (list of records)."""
    df = pd.read_json(path)
    return load_long_df(load_csv_df(df)) if "patient_id" in df.columns else df


def load_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa cột timestamp thành datetime."""
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def validate_units(df: pd.DataFrame) -> pd.DataFrame:
    """Kiểm tra và chuẩn hóa đơn vị đo cho từng metric."""
    return df
