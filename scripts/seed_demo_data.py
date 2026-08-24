"""Seed dữ liệu demo cho UI (bảng Bản ghi cá nhân).

Nguồn:
  1. data/sample_long.csv -> P001..P005 (lịch sử dài khác nhau: 60 → 1 ngày).
  2. Hai ca tổng hợp từ NHANES (DEMO_HYPERTENSIVE / DEMO_DIABETIC): 45 ngày
     với xu hướng xấu dần để thử Tầng 1 (z-score) và cảnh báo.

Chạy: python scripts/seed_demo_data.py [--force]
Idempotent: mặc định bỏ qua bệnh nhân đã có sẵn dữ liệu trong store.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat.store import ChatStore  # noqa: E402
from src.experiments.protocol import NHANES_FEATURES  # noqa: E402

METRIC_UNITS = {
    "systolic_bp": "mmHg", "diastolic_bp": "mmHg", "heart_rate": "bpm",
    "glucose": "mg/dL", "glucose_fasting": "mmol/L", "hba1c": "%",
    "creatinine": "mg/dL", "egfr": "mL/min", "spo2": "%", "bmi": "kg/m²",
}


def seed_from_sample_csv(store: ChatStore, force: bool) -> list[str]:
    df = pd.read_csv("data/sample_long.csv")
    seeded = []
    for pid, g in df.groupby("patient_id"):
        if not force and store.list_rows(pid):
            continue
        for r in g.itertuples(index=False):
            store.upsert(pid, str(r.timestamp), r.metric, float(r.value),
                         unit=METRIC_UNITS.get(r.metric))
        seeded.append(str(pid))
    return seeded


def _synth_patient(rng: np.random.Generator, start: dict[str, float],
                   drift: dict[str, float], days: int,
                   dates: pd.DatetimeIndex,
                   spike: dict[str, float] | None = None) -> list[dict]:
    rows = []
    spike_start = days - 7
    for i in range(days):
        t = dates[i].strftime("%Y-%m-%d")
        bump = spike if (spike and i >= spike_start) else {}
        for m in NHANES_FEATURES:
            v = start[m] + drift[m] * i + bump.get(m, 0.0) + rng.normal(0, abs(start[m]) * 0.03)
            rows.append({"timestamp": t, "metric": m, "value": round(float(v), 1),
                         "unit": METRIC_UNITS.get(m)})
    return rows


def seed_synthetic(store: ChatStore, force: bool) -> list[str]:
    """Hai ca tổng hợp: tăng huyết áp tiến triển & đái tháo đường tiến triển."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-07-10", periods=45, freq="D")
    cases = {
        "DEMO_HYPERTENSIVE": (
            {"systolic_bp": 138.0, "diastolic_bp": 84.0, "heart_rate": 76.0,
             "glucose_fasting": 5.4, "hba1c": 5.6, "creatinine": 0.9, "bmi": 26.0},
            {"systolic_bp": 0.55, "diastolic_bp": 0.25, "heart_rate": 0.1,
             "glucose_fasting": 0.004, "hba1c": 0.003, "creatinine": 0.002, "bmi": 0.02},
            {"systolic_bp": 22.0, "diastolic_bp": 10.0, "heart_rate": 8.0,
             "glucose_fasting": 0.2, "hba1c": 0.05, "creatinine": 0.03, "bmi": 0.3},
        ),
        "DEMO_DIABETIC": (
            {"systolic_bp": 126.0, "diastolic_bp": 78.0, "heart_rate": 80.0,
             "glucose_fasting": 5.9, "hba1c": 6.0, "creatinine": 1.0, "bmi": 28.5},
            {"systolic_bp": 0.08, "diastolic_bp": 0.05, "heart_rate": 0.05,
             "glucose_fasting": 0.05, "hba1c": 0.02, "creatinine": 0.004, "bmi": 0.04},
            {"glucose_fasting": 1.8, "hba1c": 0.4, "bmi": 0.8, "heart_rate": 6.0,
             "systolic_bp": 4.0, "diastolic_bp": 2.0, "creatinine": 0.02},
        ),
    }
    seeded = []
    for pid, (start, drift, spike) in cases.items():
        if not force and store.list_rows(pid):
            continue
        rng = np.random.default_rng(abs(hash(pid)) % 2**32)
        for r in _synth_patient(rng, start, drift, len(dates), dates, spike):
            store.upsert(pid, r["timestamp"], r["metric"], r["value"], unit=r["unit"])
        seeded.append(pid)
    return seeded


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed dữ liệu demo vào ChatStore")
    ap.add_argument("--force", action="store_true",
                    help="Ghi đè cả khi bệnh nhân đã có dữ liệu")
    args = ap.parse_args()

    store = ChatStore()
    p1 = seed_from_sample_csv(store, args.force)
    p2 = seed_synthetic(store, args.force)
    all_p = sorted(store.list_patients())
    print(f"Đã seed: {p1 + p2 or '(bỏ qua — đã có dữ liệu, dùng --force để ghi đè)'}")
    print(f"Bệnh nhân trong store: {all_p}")
    for pid in all_p:
        n = len(store.list_rows(pid))
        dates = len({r["timestamp"] for r in store.list_rows(pid)})
        print(f"  {pid}: {n} ô · {dates} ngày")


if __name__ == "__main__":
    main()
