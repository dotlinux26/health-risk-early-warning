"""Phát hiện bất thường theo SAI SỐ DỰ BÁO chuỗi thời gian.

Ý tưởng: không so với trung bình lịch sử (Z-Score tĩnh) mà so với GIÁ TRỊ
DỰ BÁO một bước — nếu giá trị thực lệch khỏi dự báo (residual) vượt quá độ
biến thiên bình thường của residual thì xem là bất thường kiểu "đột biến".

Backend:
    - "ewma"  : EWMA một bước (offline, không cần cài thêm gì).
    - "chronos": Amazon Chronos (mô hình nền tảng cho dự báo) nếu được cài
      (pip install chronos-forecasting). Tự động fallback về EWMA khi không có.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config


def _ewma_forecast_residuals(
    series: pd.Series,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Một-bước-dự-báo bằng EWMA. Trả (forecasts, residuals).

    forecast[t] = ewma của các giá trị TRƯỚC t (không dùng x[t]).
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    ewma = np.empty(n)
    ewma[0] = values[0]
    forecasts = np.full(n, np.nan)
    residuals = np.full(n, np.nan)
    for t in range(1, n):
        forecasts[t] = ewma[t - 1]
        residuals[t] = values[t] - forecasts[t]
        ewma[t] = alpha * values[t] + (1 - alpha) * ewma[t - 1]
    return forecasts, residuals


def ewma_forecast_zscore(series: pd.Series, alpha: float = 0.3) -> tuple[float, float]:
    """Trả (z_score của residual mới nhất, forecast mới nhất).

    z = residual_last / std(residual) — độ lệch giá trị thực so với dự báo,
    theo đơn vị độ biến thiên residual bình thường.
    """
    if series.dropna().shape[0] < 8:
        return 0.0, float("nan")
    s = series.dropna()
    _, residuals = _ewma_forecast_residuals(s, alpha)
    resid = residuals[~np.isnan(residuals)]
    if len(resid) < 5:
        return 0.0, float("nan")
    std = float(np.std(resid))
    if std == 0 or np.isnan(std):
        return 0.0, float("nan")
    z = float(resid[-1] / std)
    return z, float(s.iloc[-1] - resid[-1])


def _try_chronos(series: pd.Series) -> float | None:
    """Dự báo một bước bằng Amazon Chronos (nếu cài). None nếu không dùng được."""
    try:
        import torch  # noqa: F401
        from chronos import ChronosPipeline
    except ImportError:
        return None
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32
        )
        context = torch.tensor(series.dropna().to_numpy(dtype=float))
        forecast, _ = pipeline.predict(context=context, prediction_length=1)
        return float(forecast[0].mean().item())
    except Exception:
        return None


def detect_forecast_anomaly(
    df: pd.DataFrame,
    value_cols: list[str],
    config: Config = Config(),
) -> dict[str, dict]:
    """Trả {metric: {"z": float, "forecast": float, "flagged": bool}}."""
    out: dict[str, dict] = {}
    for col in value_cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.notna().sum() < 8:
            continue
        if config.forecast_backend == "chronos":
            fc = _try_chronos(s)
            if fc is not None:
                resid = float(s.dropna().iloc[-1]) - fc
                std = float(s.dropna().diff().std()) or 1e-9
                z = resid / std
                out[col] = {"z": round(z, 2), "forecast": round(fc, 2),
                            "flagged": abs(z) >= config.forecast_z_threshold}
                continue
        z, fc = ewma_forecast_zscore(s, alpha=config.forecast_alpha)
        out[col] = {"z": round(z, 2), "forecast": round(fc, 2) if not np.isnan(fc) else None,
                    "flagged": abs(z) >= config.forecast_z_threshold}
    return out
