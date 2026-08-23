"""Lưu trữ tích lũy dữ liệu chat theo bệnh nhân (JSONL).

Mỗi bệnh nhân một file: data/chat/{patient_id}.jsonl
Mỗi dòng là một bản ghi chuẩn: {"timestamp": "YYYY-MM-DD", "metric", "value", "unit"}
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src.ingest.parsers import ParsedRecord

DEFAULT_DIR = Path("data/chat")


class ChatStore:
    """Bộ tích lũy: append bản ghi, load về DataFrame, thống kê trạng thái."""

    def __init__(self, base_dir: str | Path = DEFAULT_DIR) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, patient_id: str) -> Path:
        return self.base_dir / f"{patient_id}.jsonl"

    def append(self, patient_id: str, records: list[ParsedRecord]) -> int:
        """Ghi thêm bản ghi; không trùng (metric, date). Trả số bản ghi mới."""
        path = self._path(patient_id)
        existing = self._load_rows(patient_id)
        seen = {(r["metric"], r["timestamp"]) for r in existing}

        added = 0
        with path.open("a", encoding="utf-8") as f:
            for r in records:
                key = (r.metric, r.date.isoformat())
                if key in seen:
                    continue
                line = {
                    "timestamp": r.date.isoformat(),
                    "metric": r.metric,
                    "value": r.value,
                    "unit": r.unit,
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                seen.add(key)
                added += 1
        return added

    def _load_rows(self, patient_id: str) -> list[dict]:
        path = self._path(patient_id)
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def load(self, patient_id: str) -> pd.DataFrame:
        """Trả DataFrame long-format chuẩn (patient_id|timestamp|metric|value|unit)."""
        rows = self._load_rows(patient_id)
        df = pd.DataFrame(
            {
                "patient_id": patient_id,
                "timestamp": [r["timestamp"] for r in rows],
                "metric": [r["metric"] for r in rows],
                "value": [r["value"] for r in rows],
                "unit": [r.get("unit", "") for r in rows],
            }
        )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def status(self, patient_id: str, min_points: int = 7) -> dict:
        """Thống kê: số ngày đo, các chỉ số, đã đủ đánh giá chuỗi thời gian chưa."""
        df = self.load(patient_id)
        if df.empty:
            return {
                "patient_id": patient_id,
                "has_data": False,
                "unique_dates": 0,
                "needed_dates": min_points,
                "ready": False,
                "metrics": [],
                "total_records": 0,
            }
        by_metric = df.groupby("metric")["timestamp"].apply(lambda s: s.dt.date.unique().tolist())
        return {
            "patient_id": patient_id,
            "has_data": True,
            "unique_dates": int(df["timestamp"].dt.date.nunique()),
            "needed_dates": min_points,
            "ready": int(df["timestamp"].dt.date.nunique()) >= min_points,
            "metrics": [
                {"metric": m, "days": len(v), "dates": [d.isoformat() for d in v]}
                for m, v in by_metric.items()
            ],
            "total_records": len(df),
        }

    def reset(self, patient_id: str) -> None:
        path = self._path(patient_id)
        if path.exists():
            path.unlink()

    def list_patients(self) -> list[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.jsonl"))

    # ------------------------------------------------------------------ #
    # P1 — Quản lý bản ghi cá nhân theo ngày (docs/15 U10)
    # ------------------------------------------------------------------ #
    def _write_rows(self, patient_id: str, rows: list[dict]) -> None:
        path = self._path(patient_id)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def upsert(
        self, patient_id: str, timestamp: str, metric: str,
        value: float | None, unit: str = "",
    ) -> dict:
        """Thêm/sửa một ô dữ liệu (metric, ngày). value=None -> xóa giá trị."""
        from datetime import datetime as _dt

        _dt.strptime(timestamp, "%Y-%m-%d")  # validate
        rows = self._load_rows(patient_id)
        key = (metric, timestamp)
        found = False
        out = []
        for r in rows:
            if (r["metric"], r["timestamp"]) == key:
                if value is None:
                    continue  # xóa dòng
                r = {**r, "value": float(value), "unit": unit or r.get("unit", "")}
                found = True
            out.append(r)
        if not found and value is not None:
            out.append({"timestamp": timestamp, "metric": metric,
                        "value": float(value), "unit": unit})
        self._write_rows(patient_id, out)
        return {"patient_id": patient_id, "timestamp": timestamp,
                "metric": metric, "value": value}

    def delete_value(self, patient_id: str, timestamp: str, metric: str) -> None:
        """Xóa một ô (metric, ngày); ném ValueError nếu không tồn tại."""
        rows = self._load_rows(patient_id)
        kept = [r for r in rows
                if not (r["metric"] == metric and r["timestamp"] == timestamp)]
        if len(kept) == len(rows):
            raise ValueError(f"Không tìm thấy ({metric}, {timestamp}).")
        self._write_rows(patient_id, kept)

    def table_by_date(self, patient_id: str) -> dict:
        """Bảng theo ngày: mỗi hàng một ngày, cột là các chỉ số (U10)."""
        rows = self._load_rows(patient_id)
        dates = sorted({r["timestamp"] for r in rows}, reverse=True)
        metrics = sorted({r["metric"] for r in rows})
        grid = {}
        for d in dates:
            day_rows = [r for r in rows if r["timestamp"] == d]
            grid[d] = {
                r["metric"]: {"value": r.get("value"),
                              "unit": r.get("unit", "")}
                for r in day_rows
            }
        return {"patient_id": patient_id, "metrics": metrics,
                "dates": dates, "grid": grid}
