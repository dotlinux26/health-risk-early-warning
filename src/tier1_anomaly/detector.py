"""Bộ điều phối Tầng 1 — gộp Z-Score + Isolation Forest + xu hướng."""
from __future__ import annotations

import pandas as pd

from src.config import Config
from src.tier1_anomaly import AnomalyRecord
from src.tier1_anomaly.isolation_forest import detect_isolation_forest
from src.tier1_anomaly.trend import detect_ewma_crossing
from src.tier1_anomaly.zscore import detect_zscore


def run_tier1(
    df: pd.DataFrame,
    value_cols: list[str],
    config: Config = Config(),
) -> list[AnomalyRecord]:
    """Chạy toàn bộ Tầng 1 cho một bệnh nhân (wide format đã có baseline).

    - Z-Score cá nhân hóa: gắn cờ bất thường.
    - Isolation Forest: gắn cờ thay đổi đồng thời nhiều chỉ số.
    - EWMA: gán nhãn xu hướng cho từng chỉ số.
    """
    if len(df) < config.min_points:
        return []

    records = detect_zscore(df, value_cols, config)
    forest_records = detect_isolation_forest(df, value_cols, config)

    # Gộp: cập nhật trạng thái flagged/trend từ Isolation Forest và EWMA
    trends = detect_ewma_crossing(df, value_cols)
    by_metric = {r.metric: r for r in records}
    for r in forest_records:
        if r.metric in by_metric:
            by_metric[r.metric].flagged = by_metric[r.metric].flagged or r.flagged
        else:
            by_metric[r.metric] = r

    for r in by_metric.values():
        if r.trend == "unknown" and r.metric in trends:
            r.trend = trends[r.metric]
        elif r.metric in trends and r.trend == "stable" and trends[r.metric] != "stable":
            r.trend = trends[r.metric]

    return sorted(by_metric.values(), key=lambda r: -abs(r.z_score or 0.0))


def tier1_summary(records: list[AnomalyRecord]) -> dict:
    """Tóm tắt Tầng 1 để đưa vào báo cáo."""
    flagged = [r for r in records if r.flagged]
    return {
        "total_metrics": len(records),
        "flagged_metrics": [r.metric for r in flagged],
        "max_zscore": max((abs(r.z_score or 0) for r in records), default=0.0),
    }
