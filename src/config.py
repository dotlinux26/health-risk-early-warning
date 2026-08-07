"""Cấu hình trung tâm của hệ thống."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Các tham số toàn cục cho pipeline 3 tầng."""

    # Dữ liệu
    data_dir: Path = Path("data")
    min_points: int = 7                 # < 7 điểm -> INSUFFICIENT_DATA
    missing_ratio_limit: float = 0.30   # > 30% thiếu -> không nội suy, chỉ báo missing

    # Tầng 1
    zscore_window_days: int = 90        # cửa sổ đường cơ sở cá nhân
    zscore_threshold: float = 2.0       # |Z| >= ngưỡng -> bất thường
    isolation_forest_contamination: float = 0.05
    isolation_forest_metric_window: int = 30
    ewma_lambda: float = 0.2

    # Tầng 3
    risk_weights: dict[str, float] = field(default_factory=lambda: {
        "stat": 0.30,      # trọng số bất thường thống kê (Tầng 1)
        "knowledge": 0.35, # trọng số tri thức y khoa (Tầng 2)
        "ml": 0.25,        # trọng số mô hình học máy
        "trend": 0.10,     # trọng số xu hướng dài hạn
    })
    risk_level_thresholds: tuple[float, float] = (0.33, 0.66)  # Low/Medium/High
    # An toàn lâm sàng: nếu có luật nghiêm trọng (severity >= ngưỡng này) kích hoạt
    # thì điểm rủi ro ít nhất được nâng lên mức TRUNG_BINH (floor).
    critical_rule_severity: float = 0.7
    critical_rule_floor: float = 0.5

    # Mô hình
    model_name: str = "lightgbm"
    random_state: int = 42
    use_ml: bool = True                     # bật điểm ML (LightGBM) trong Tầng 3
    ml_model_path: Path = Path("data/models/risk_lgbm_real.joblib")
    ml_min_days: int = 1                    # cần ít nhất bấy nhiêu ngày mới chạy ML

    # Phát hiện bất thường theo sai số dự báo (Tầng 1, tùy chọn)
    forecast_backend: str = "ewma"          # "ewma" (offline) | "chronos" (nếu cài)
    forecast_alpha: float = 0.3
    forecast_z_threshold: float = 2.5

    # Đầu ra
    output_dir: Path = Path("report")


CONFIG = Config()
