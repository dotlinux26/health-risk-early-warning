"""Parser tin nhắn chat hàng ngày -> bản ghi chỉ số chuẩn.

Hỗ trợ thêm dạng huyết áp "148/92" (tâm thu/tâm trương) mà parser file không có:
    "Huyết áp 148/92"  -> systolic_bp=148, diastolic_bp=92
    "HA 138/85"        -> như trên
Ngoài ra tái sử dụng MetricParser (patterns.json) cho các chỉ số khác.
"""
from __future__ import annotations

import re
from datetime import date

from src.ingest.parsers import DATE_RE, MetricParser, ParsedRecord, _norm_number, _parse_date

BP_PAIR_RE = re.compile(
    r"(?:huyết\s*áp|huyet\s*ap|ha|bp|blood\s*pressure)\s*[:=]?\s*"
    r"(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)

# Chỉ số ưu tiên nhắc khi thiếu dữ liệu
RECOMMENDED_METRICS = [
    ("systolic_bp", "huyết áp (vd: Huyết áp 125/80)"),
    ("glucose_fasting", "đường huyết lúc đói"),
    ("hba1c", "HbA1c (nếu có xét nghiệm)"),
    ("heart_rate", "nhịp tim"),
    ("weight", "cân nặng"),
]


class ChatParser:
    """Chuyển một dòng nhật ký sức khỏe thành các ParsedRecord."""

    def __init__(self) -> None:
        self.metric_parser = MetricParser()

    def parse(self, text: str, day: date | None = None) -> list[ParsedRecord]:
        if day is None:
            # Nhận diện ngày trong tin nhắn, vd "ngày 10/06/2025: Huyết áp 135/85"
            day = _parse_date(text) if DATE_RE.search(text) else date.today()
        records: list[ParsedRecord] = []

        # 1) Cặp huyết áp dạng 148/92 (có thể lồng trong câu)
        cleaned = BP_PAIR_RE.sub(" ", text)
        for m in BP_PAIR_RE.finditer(text):
            sys = _norm_number(m.group(1))
            dia = _norm_number(m.group(2))
            records.append(ParsedRecord("systolic_bp", sys, "mmHg", day))
            records.append(ParsedRecord("diastolic_bp", dia, "mmHg", day))

        # 2) Các chỉ số còn lại (alias song ngữ)
        records.extend(self.metric_parser.parse_lines([cleaned]))

        # 3) Loại trùng (metric + ngày)
        seen: dict[tuple[str, date], ParsedRecord] = {}
        for r in records:
            seen[(r.metric, r.date)] = r
        return list(seen.values())
