"""Sinh dữ liệu tổng hợp có NHÃN để huấn luyện mô hình ML Tầng 3.

Bài toán "cảnh báo sớm":
    - Mỗi bệnh nhân có một trạng thái ẩn (khỏe mạnh / tăng huyết áp / đái tháo
      đường / suy thận). Trạng thái này điều khiển đường cong chỉ số theo thời gian.
    - Đặc trưng (X) lấy từ 60 ngày đầu.
    - Nhãn (y) = 1 nếu có SỰ KIỆN NGƯỠNG xảy ra trong các ngày 61–90 (chỉ số
      vượt ngưỡng lâm sàng và DUY TRÌ >= 3 ngày liên tiếp).

Mô hình học phát hiện sớm (từ xu hướng trong 60 ngày đầu dự đoán sự kiện sắp
tới) chứ không chỉ nhìn giá trị tuyệt đối. Sinh này có kiểm soát seed nên tái
lập được, phục vụ huấn luyện/đánh giá pipeline trước khi có dữ liệu thực.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class MetricSpec:
    """Cấu hình một chỉ số trong một trạng thái bệnh."""

    base: float          # giá trị khởi điểm
    final_add: float     # mức tăng tối đa sau ramp
    ramp_t0: float       # điểm uốn của đường logistic
    ramp_w: float        # độ dốc của đường logistic
    amplitude: float     # biên độ chu kỳ tuần
    noise: float         # độ lệch chuẩn nhiễu ngày-ngày


@dataclass
class Condition:
    """Định nghĩa một trạng thái bệnh ẩn."""

    name: str
    metrics: dict[str, MetricSpec]
    critical: list[str]          # chỉ số dùng để xác định "sự kiện ngưỡng"
    thresholds: dict[str, float] # ngưỡng sự kiện theo chỉ số


# --- Tham số sinh theo trạng thái ------------------------------------------

def _ramp(t: np.ndarray, spec: MetricSpec) -> np.ndarray:
    """Đường cong logistic: base -> base + final_add, uốn tại ramp_t0."""
    return spec.base + spec.final_add / (1.0 + np.exp(-(t - spec.ramp_t0) / spec.ramp_w))


def _series(t: np.ndarray, spec: MetricSpec, rng: np.random.Generator) -> np.ndarray:
    drift = _ramp(t, spec)
    seasonal = spec.amplitude * np.sin(2 * np.pi * t / 7.0)
    noise = rng.normal(0, spec.noise, len(t))
    return np.maximum(drift + seasonal + noise, 0.0)


CONDITIONS: dict[str, Condition] = {
    "healthy": Condition(
        name="healthy",
        metrics={
            "systolic_bp": MetricSpec(118, 4, 120, 30, 2.0, 3.5),
            "diastolic_bp": MetricSpec(76, 2, 120, 30, 1.2, 2.5),
            "heart_rate": MetricSpec(70, 2, 120, 30, 1.5, 4.0),
            "glucose_fasting": MetricSpec(5.0, 0.2, 120, 30, 0.05, 0.25),
            "hba1c": MetricSpec(5.1, 0.1, 120, 30, 0.0, 0.15),
            "creatinine": MetricSpec(0.85, 0.05, 120, 30, 0.0, 0.05),
            "egfr": MetricSpec(100, -3, 120, 30, 0.0, 3.0),
            "spo2": MetricSpec(97.5, -0.3, 120, 30, 0.0, 0.6),
            "bmi": MetricSpec(23.0, 0.3, 120, 30, 0.0, 0.2),
        },
        critical=[],
        thresholds={},
    ),
    "hypertension": Condition(
        name="hypertension",
        metrics={
            "systolic_bp": MetricSpec(122, 40, 45, 12, 2.0, 3.5),
            "diastolic_bp": MetricSpec(80, 22, 45, 12, 1.2, 2.5),
            "heart_rate": MetricSpec(72, 4, 45, 12, 1.5, 4.0),
            "glucose_fasting": MetricSpec(5.2, 0.3, 45, 12, 0.05, 0.25),
            "hba1c": MetricSpec(5.2, 0.2, 45, 12, 0.0, 0.15),
            "creatinine": MetricSpec(0.9, 0.1, 45, 12, 0.0, 0.05),
            "egfr": MetricSpec(95, -8, 45, 12, 0.0, 3.0),
            "spo2": MetricSpec(97.0, -0.5, 45, 12, 0.0, 0.6),
            "bmi": MetricSpec(25.0, 1.5, 45, 12, 0.0, 0.2),
        },
        critical=["systolic_bp", "diastolic_bp"],
        thresholds={"systolic_bp": 140.0, "diastolic_bp": 90.0},
    ),
    "diabetes": Condition(
        name="diabetes",
        metrics={
            "systolic_bp": MetricSpec(122, 12, 50, 12, 2.0, 3.5),
            "diastolic_bp": MetricSpec(78, 6, 50, 12, 1.2, 2.5),
            "heart_rate": MetricSpec(74, 6, 50, 12, 1.5, 4.0),
            "glucose_fasting": MetricSpec(5.4, 2.6, 45, 12, 0.05, 0.25),
            "hba1c": MetricSpec(5.4, 2.4, 45, 12, 0.0, 0.15),
            "creatinine": MetricSpec(0.9, 0.15, 50, 12, 0.0, 0.05),
            "egfr": MetricSpec(94, -10, 50, 12, 0.0, 3.0),
            "spo2": MetricSpec(97.0, -0.5, 50, 12, 0.0, 0.6),
            "bmi": MetricSpec(27.0, 2.0, 50, 12, 0.0, 0.2),
        },
        critical=["glucose_fasting", "hba1c"],
        thresholds={"glucose_fasting": 7.0, "hba1c": 6.5},
    ),
    "kidney": Condition(
        name="kidney",
        metrics={
            "systolic_bp": MetricSpec(125, 18, 50, 12, 2.0, 3.5),
            "diastolic_bp": MetricSpec(80, 10, 50, 12, 1.2, 2.5),
            "heart_rate": MetricSpec(72, 4, 50, 12, 1.5, 4.0),
            "glucose_fasting": MetricSpec(5.3, 0.4, 50, 12, 0.05, 0.25),
            "hba1c": MetricSpec(5.3, 0.3, 50, 12, 0.0, 0.15),
            "creatinine": MetricSpec(0.9, 0.9, 42, 12, 0.0, 0.05),
            "egfr": MetricSpec(95, -38, 42, 12, 0.0, 3.0),
            "spo2": MetricSpec(96.5, -1.0, 50, 12, 0.0, 0.6),
            "bmi": MetricSpec(24.0, 0.8, 50, 12, 0.0, 0.2),
        },
        critical=["creatinine", "egfr"],
        thresholds={"creatinine": 1.4, "egfr": 60.0},
    ),
}

# Nhãn trạng thái dùng để thống kê (không dùng trực tiếp làm nhãn ML)
STATE_LABEL: dict[str, str] = {
    "healthy": "Khỏe mạnh",
    "hypertension": "Tăng huyết áp",
    "diabetes": "Đái tháo đường",
    "kidney": "Suy thận",
}


def gen_patient(
    state: str,
    days: int = 90,
    seed: int | None = None,
    missing_ratio: float = 0.05,
    event_shift: float = 0.0,
) -> pd.DataFrame:
    """Sinh dữ liệu ngày cho một bệnh nhân (wide format, cột timestamp).

    event_shift (ngày) làm chậm đường ramp -> tạo ra các ca bệnh CHƯA vượt
    ngưỡng trong cửa sổ quan sát (nhãn 0) dù đang tiến triển.
    """
    cond = CONDITIONS[state]
    rng = np.random.default_rng(seed)
    t = np.arange(days, dtype=float)

    rows: dict[str, np.ndarray] = {"timestamp": pd.date_range("2025-01-01", periods=days, freq="D").values.astype("datetime64[ns]")}
    for metric, spec in cond.metrics.items():
        shifted = MetricSpec(
            base=spec.base,
            final_add=spec.final_add,
            ramp_t0=spec.ramp_t0 + event_shift,
            ramp_w=spec.ramp_w,
            amplitude=spec.amplitude,
            noise=spec.noise,
        )
        rows[metric] = _series(t, shifted, rng)

    df = pd.DataFrame(rows)

    # Nhiễu phi lâm sàng: thiếu giá trị rải rác (như bệnh nhân quên đo)
    rng2 = np.random.default_rng(seed + 1 if seed is not None else None)
    metric_cols = [c for c in df.columns if c != "timestamp"]
    mask = rng2.random((len(df), len(metric_cols))) < missing_ratio
    df[metric_cols] = df[metric_cols].mask(mask)

    return df


def event_days(df: pd.DataFrame, state: str) -> int | None:
    """Trả về ngày đầu tiên có sự kiện ngưỡng (None nếu không có)."""
    cond = CONDITIONS[state]
    if not cond.critical:
        return None
    sustained: dict[str, np.ndarray] = {}
    for metric in cond.critical:
        thr = cond.thresholds[metric]
        s = df[metric].to_numpy(dtype=float)
        above = s >= thr
        # duy trì 3 ngày liên tiếp
        triple = np.convolve(above.astype(int), np.ones(3, dtype=int), mode="valid") == 3
        sustained[metric] = triple
    # ngày có sự kiện = ngày đầu tiên bất kỳ chỉ số nào duy trì >= 3 ngày
    merged = np.zeros(len(df) - 2, dtype=bool)
    for arr in sustained.values():
        merged |= arr
    hits = np.where(merged)[0]
    return int(hits[0] + 2) if len(hits) else None


def generate_labeled_dataset(
    n_per_condition: int = 60,
    days: int = 90,
    feature_window: int = 60,
    event_shift_max: float = 35.0,
    seed: int = 42,
) -> tuple[list[pd.DataFrame], list[int], list[str]]:
    """Sinh dataset: (wide_dfs, labels, states).

    Nhãn y = 1 nếu sự kiện xảy ra trong (feature_window, days).
    Nhãn 0 gồm: người khỏe + người bệnh tiến triển chậm (chưa tới ngưỡng).
    """
    rng = np.random.default_rng(seed)
    wide_dfs: list[pd.DataFrame] = []
    labels: list[int] = []
    states: list[str] = []

    i = 0
    for state in CONDITIONS:
        for _ in range(n_per_condition):
            shift = rng.uniform(0, event_shift_max)
            df = gen_patient(state, days=days, seed=(seed + i), event_shift=shift)
            ev = event_days(df, state)
            y = 1 if (ev is not None and feature_window <= ev < days) else 0
            wide_dfs.append(df)
            labels.append(y)
            states.append(state)
            i += 1

    return wide_dfs, labels, states


def build_training_frame(
    wide_df: pd.DataFrame,
    feature_window: int = 60,
    value_cols: list[str] | None = None,
    config=None,
) -> pd.DataFrame:
    """Chuyển 60 ngày đầu của một bệnh nhân thành một dòng đặc trưng.

    Dòng đặc trưng lấy từ ROW CUỐI của build_feature_matrix trên cửa sổ 60 ngày
    (chỉ dùng dữ liệu quá khứ, không rò rỉ tương lai).
    """
    from src.config import Config
    from src.data.features import build_feature_matrix

    cfg = config or Config()
    vc = value_cols or [c for c in wide_df.columns if c != "timestamp"]
    seg = wide_df.head(feature_window).copy()
    fm = build_feature_matrix(seg, cfg, value_cols=vc)
    fm = fm.drop(columns=[c for c in ("timestamp", "patient_id") if c in fm.columns])
    return fm.tail(1).reset_index(drop=True)


def build_train_matrix(
    wide_dfs: list[pd.DataFrame],
    feature_window: int = 60,
    value_cols: list[str] | None = None,
    config=None,
) -> pd.DataFrame:
    """Gộp toàn bộ bệnh nhân thành ma trận X (mỗi bệnh nhân một dòng)."""
    frames = [build_training_frame(d, feature_window, value_cols, config) for d in wide_dfs]
    return pd.concat(frames, ignore_index=True)
