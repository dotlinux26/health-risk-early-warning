"""Parser văn bản -> bản ghi chỉ số sức khỏe (regex, không cần LLM).

Nhận diện các dạng:
    "Huyết áp tâm thu: 128 mmHg"
    "Glucose máu 7.2 mmol/L"
    "Systolic BP = 128"
    "Ngày 05/03/2025 ... Creatinine 1.1"
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

DEFAULT_PATTERNS = Path(__file__).parent / "patterns.json"

DATE_RE = re.compile(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})")
VALUE_RE = re.compile(r"[-+]?\d{1,4}(?:[.,]\d{1,3})?")
UNIT_RE = r"(?:mmol/L|mg/dL|mmHg|kg/m2|mL/min/1.73m2|bpm|cm|kg|%)?"

METRIC_UNITS: dict[str, str] = {
    "systolic_bp": "mmHg", "diastolic_bp": "mmHg", "heart_rate": "bpm",
    "glucose": "mmol/L", "glucose_fasting": "mmol/L", "hba1c": "%",
    "creatinine": "mg/dL", "egfr": "mL/min/1.73m2", "spo2": "%",
    "bmi": "kg/m2", "weight": "kg", "height": "cm",
}


@dataclass
class ParsedRecord:
    metric: str
    value: float
    unit: str
    date: date


def _norm_number(s: str) -> float:
    return float(s.replace(",", "."))


def _parse_date(s: str) -> date:
    m = DATE_RE.search(s)
    if not m:
        return date.today()
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    try:
        return date(y, mo, d)
    except ValueError:
        return date.today()


class MetricParser:
    """Regex-based extractor: alias + giá trị (+ unit)."""

    def __init__(self, patterns_path: str | Path = DEFAULT_PATTERNS) -> None:
        with open(patterns_path, encoding="utf-8") as f:
            self.patterns: dict = json.load(f)
        # Sắp xếp alias theo độ dài giảm dần để khớp cụm dài trước (vd: "huyết áp tâm thu"
        # trước "huyết áp"), tránh khớp sai.
        self._regexes: list[tuple[str, str, re.Pattern]] = []
        for metric, cfg in self.patterns.items():
            unit = cfg.get("unit") or METRIC_UNITS.get(metric, "")
            for alias in sorted(cfg["aliases"], key=len, reverse=True):
                esc = re.escape(alias)
                # sep cho phép ": ", " = ", " | ", "-", tab... giữa alias và giá trị
                sep = r"[\s:=|\-,–—]*"
                pat = re.compile(
                    rf"{esc}{sep}(?P<value>{VALUE_RE.pattern}){sep}(?P<unit>{UNIT_RE})",
                    re.IGNORECASE,
                )
                self._regexes.append((metric, unit, pat))

    def parse_lines(self, lines: list[str]) -> list[ParsedRecord]:
        records: list[ParsedRecord] = []
        report_date = date.today()
        for line in lines:
            d = _parse_date(line)
            # Ngày gần nhất xuất hiện trước dòng chỉ số được ưu tiên
            if re.search(DATE_RE, line):
                report_date = d
            for metric, default_unit, pat in self._regexes:
                for m in pat.finditer(line):
                    value = _norm_number(m.group("value"))
                    unit = m.group("unit").strip() or default_unit
                    if unit == "mmol/L" and metric in {"glucose", "glucose_fasting"}:
                        pass  # giữ nguyên (chuẩn hóa sau nếu cần)
                    records.append(
                        ParsedRecord(
                            metric=metric,
                            value=round(value, 3),
                            unit=unit,
                            date=report_date,
                        )
                    )
        return records
