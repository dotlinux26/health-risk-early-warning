"""Điểm vào CLI của pipeline 3 tầng.

Ví dụ chạy:
    python -m src.main --input data/sample.csv --output report/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CONFIG
from src.core.pipeline import (
    VALUE_COLUMNS,
    RiskResult,
    assess_patient,
    save_report,
)
from src.data.loader import load_csv, load_long_df
from src.tier2_knowledge.rules import KnowledgeBase
from src.tier3_risk.scoring import RiskScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Hệ thống cảnh báo sớm nguy cơ sức khỏe 3 tầng")
    parser.add_argument("--input", type=str, required=True, help="CSV đầu vào (schema chuẩn)")
    parser.add_argument("--output", type=str, default="report", help="Thư mục báo cáo")
    parser.add_argument("--json-out", type=str, default=None, help="Xuất toàn bộ ra file JSON")
    args = parser.parse_args()

    config = CONFIG
    scorer = RiskScorer(config, kb=KnowledgeBase())
    df_raw = load_csv(args.input)
    patients = load_long_df(df_raw)

    out_dir = Path(args.output)
    all_results: dict[str, dict] = {}

    for pid, wide in patients.items():
        result = assess_patient(wide, config, scorer)
        all_results[pid] = result
        if result["risk_level"] == "INSUFFICIENT_DATA":
            print(f"[{pid}] {result['message']}")
            continue

        # Tạo object RiskResult để xuất báo cáo markdown chuẩn
        risk_result = RiskResult(
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            affected_systems=result["affected_systems"],
            evidence=result["evidence"],
            recommendations=result["recommendations"],
            components=result["components"],
            metrics_detail=result["metrics_detail"],
        )
        path = save_report(pid, risk_result, out_dir)
        print(f"[{pid}] Rủi ro: {result['risk_level']} (điểm {result['risk_score']}) -> {path}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON đầy đủ: {args.json_out}")


if __name__ == "__main__":
    main()
