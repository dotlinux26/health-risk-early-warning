"""Tầng 4 — Giải thích tự nhiên bằng mô hình ngôn ngữ (LLM).

Tầng này là TUỲ CHỌN và PLUGGABLE: hệ thống chạy đầy đủ 3 tầng khi nó vắng
mặt. Có nó thì thêm lời giải thích tự nhiên, không có nó thì vẫn ra báo cáo.

Vị trí trong luồng (sau FUSION, xem notes/02):
    Patient Features → ML → FUSION → Final Risk → Evidence → Explainer → text

Nguyên tắc: tầng 4 KHÔNG quyết định nguy cơ, chỉ giải thích dựa trên bộ bằng
chứng đã có (context). Đánh giá bằng Faithfulness / Completeness / Consistency /
Hallucination, KHÔNG dùng ROC-AUC.
"""
from __future__ import annotations

from typing import Any, Protocol


class Explainer(Protocol):
    """Interface tầng 4 — mọi model giải thích (80–100M, LLM API...) phải thoả."""

    def explain(self, context: dict[str, Any]) -> str:
        """Sinh lời giải thích tự nhiên từ bộ bằng chứng fusion.

        context (do src/core/pipeline.py cung cấp):
            ml_risk: float | None        điểm mô hình ML
            stat_anomalies: list[dict]   bất thường Tầng 1
            final_risk: float            điểm rủi ro sau fusion
            level: str                   CAO | TRUNG_BINH | THAP
            features: dict               snapshot chỉ số hiện tại
            rules: list[dict]            luật Tầng 2 kích hoạt (evidence)

        Trả về: văn bản giải thích tự nhiên (tiếng Việt, ngắn gọn, trung thực
        với context — không bịa thông tin ngoài input).
        """
        ...
