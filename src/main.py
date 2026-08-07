"""Điểm vào chính của pipeline 3 tầng.

Ví dụ chạy:
    python -m src.main --input data/sample.csv --output report/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import Config, CONFIG
from src.data.features import build_feature_matrix
from src.data.loader import load_csv, load_long_df
from src.data.preprocess import build_baseline, impute_missing, resample_to_daily
from src.tier1_anomaly.detector import run_tier1, tier1_summary
from src.tier2_knowledge.rules import KnowledgeBase
from src.tier3_risk.report import render_json, render_markdown, save_report
from src.tier3_risk.scoring import RiskScorer

VALUE_COLUMNS = [
    "systolic_bp", "diastolic_bp", "heart_rate", "glucose",
    "glucose_fasting", "hba1c", "creatinine", "egfr", "spo2", "bmi",
]


def assess_patient(
    df_wide: pd.DataFrame,
    config: Config,
    scorer: RiskScorer,
) -> dict:
    """Chạy 3 tầng cho một bệnh nhân (wide format)."""
    # 1) Làm sạch & đường cơ sở cá nhân
    df = resample_to_daily(df_wide)
    df = impute_missing(df, config)
    value_cols = [c for c in VALUE_COLUMNS if c in df.columns]
    if df.empty or not value_cols:
        return {"risk_level": "INSUFFICIENT_DATA", "message": "Không có chỉ số nào."}

    # Snapshot giá trị hiện tại (dòng cuối) — Tầng 2 luôn kích hoạt luật theo
    # giá trị hiện tại, kể cả khi chưa có lịch sử (mới chỉ 1 lần khám).
    last = df.iloc[-1]
    snapshot = {c: float(last[c]) for c in value_cols if not pd.isna(last[c])}
    if not snapshot:
        return {"risk_level": "INSUFFICIENT_DATA", "message": "Không đủ dữ liệu giá trị."}

    # 2) Tầng 1 — bất thường cá nhân hóa (cần >= min_points; nếu thiếu thì
    #    bỏ qua, hệ thống vẫn đánh giá bằng tri thức y khoa).
    records: list = []
    tier1_note: str | None = None
    if len(df) >= config.min_points:
        df = build_baseline(df, config.zscore_window_days)
        records = run_tier1(df, value_cols, config)
    else:
        tier1_note = "Chưa đủ lịch sử để phân tích chuỗi thời gian (cần ≥ 7 điểm); đánh giá theo tri thức y khoa."

    # 3) Tầng 2 — tri thức y khoa (luôn chạy từ snapshot giá trị hiện tại)
    # 4) Tầng 3 — tổng hợp rủi ro
    result = scorer.score(records, snapshot=snapshot)

    return {
        "tier1_summary": tier1_summary(records),
        "tier1_note": tier1_note,
        **result.to_dict(),
    }


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
        from src.tier3_risk.scoring import RiskResult

        risk_result = RiskResult(
            risk_level=result["risk_level"],
            risk_score=result["risk_score"],
            affected_systems=result["affected_systems"],
            evidence=result["evidence"],
            recommendations=result["recommendations"],
            components=result["components"],
        )
        path = save_report(pid, risk_result, out_dir)
        print(f"[{pid}] Rủi ro: {result['risk_level']} (điểm {result['risk_score']}) -> {path}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON đầy đủ: {args.json_out}")


if __name__ == "__main__":
    main()
