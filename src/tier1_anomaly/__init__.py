"""Tầng 1 — Phát hiện bất thường & xu hướng cá nhân hóa."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnomalyRecord:
    """Một bất thường phát hiện được ở Tầng 1.

    Đây chính là đơn vị "minh chứng dữ liệu" để truy xuất (XAI).
    """

    metric: str
    current: float
    baseline_mean: float
    z_score: float | None
    window_days: int
    direction: str          # "up" | "down"
    trend: str              # "rising" | "falling" | "stable"
    flagged: bool
    forecast_z: float | None = None   # Z của sai số dự báo (nếu bật)
    forecast_flagged: bool = False    # bất thường theo sai số dự báo

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "current": self.current,
            "baseline_mean": self.baseline_mean,
            "z_score": self.z_score,
            "window_days": self.window_days,
            "direction": self.direction,
            "trend": self.trend,
            "flagged": self.flagged,
            "forecast_z": self.forecast_z,
            "forecast_flagged": self.forecast_flagged,
        }
