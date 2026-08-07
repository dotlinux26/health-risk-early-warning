"""Pipeline ingest: file báo cáo -> dataset chuẩn (long format).

Đầu ra là DataFrame đúng schema đầu vào của pipeline chính:
    patient_id | timestamp | metric | value | unit
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.ingest.extractor import extract_text
from src.ingest.llm_extractor import extract_with_llm
from src.ingest.parsers import MetricParser, ParsedRecord


def _dedupe(records: list[ParsedRecord]) -> list[ParsedRecord]:
    """Loại bỏ trùng lặp (cùng metric + ngày, lấy lần xuất hiện cuối)."""
    seen: dict[tuple[str, date], ParsedRecord] = {}
    for r in records:
        seen[(r.metric, r.date)] = r
    return list(seen.values())


def ingest_file(
    path: str | Path,
    patient_id: str | None = None,
    use_llm: bool = False,
) -> pd.DataFrame:
    """Đọc một file PDF/DOCX/TXT và trả DataFrame chuẩn.

    - use_llm=False (mặc định): regex, chạy offline, không cần server.
    - use_llm=True: ưu tiên LLM (cần Ollama/API), fallback regex khi lỗi.
    """
    path = Path(path)
    pid = patient_id or path.stem
    lines = extract_text(path)

    records = MetricParser().parse_lines(lines)

    if use_llm:
        llm_records = extract_with_llm("\n".join(lines))
        for r in llm_records:
            metric = r.get("metric")
            value = r.get("value")
            if metric and isinstance(value, (int, float)):
                d = r.get("date") or date.today()
                try:
                    parsed_date = datetime.fromisoformat(str(d)).date()
                except ValueError:
                    parsed_date = date.today()
                records.append(
                    ParsedRecord(metric=metric, value=float(value), unit=r.get("unit", ""), date=parsed_date)
                )

    records = _dedupe(records)
    df = pd.DataFrame(
        [
            {
                "patient_id": pid,
                "timestamp": pd.Timestamp(r.date),
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
            }
            for r in records
        ]
    )
    return df


def ingest_directory(dir_path: str | Path, use_llm: bool = False) -> pd.DataFrame:
    """Ingest toàn bộ file trong thư mục -> một dataset ghép."""
    frames: list[pd.DataFrame] = []
    for f in sorted(Path(dir_path).glob("*")):
        if f.suffix.lower() in {".pdf", ".docx", ".doc", ".txt"}:
            try:
                frames.append(ingest_file(f, use_llm=use_llm))
            except Exception as exc:
                print(f"[ingest] Bỏ qua {f.name}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["patient_id", "timestamp", "metric", "value", "unit"])
    return pd.concat(frames, ignore_index=True)


def save_dataset(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Lưu dataset chuẩn ra CSV (sẵn sàng đưa vào pipeline chính)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
