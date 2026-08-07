"""Sinh báo cáo đầu ra dạng văn bản cấu trúc (markdown)."""
from __future__ import annotations

from pathlib import Path

from src.tier3_risk.scoring import RiskResult


def render_markdown(patient_id: str, result: RiskResult) -> str:
    """Xuất báo cáo markdown chuẩn hóa cho một bệnh nhân."""
    lines: list[str] = []
    lines.append(f"# Báo cáo đánh giá nguy cơ sức khỏe — Bệnh nhân {patient_id}")
    lines.append("")
    lines.append(f"- **Phân loại rủi ro:** {result.risk_level} (điểm {result.risk_score:.3f})")
    lines.append(f"- **Hệ cơ quan có khả năng ảnh hưởng:** "
                 + (", ".join(result.affected_systems) if result.affected_systems else "Chưa xác định"))
    lines.append("")
    lines.append("## Giải thích thông số (minh chứng dữ liệu)")
    if result.evidence:
        for e in result.evidence:
            lines.append(f"- {e['message']}")
    else:
        lines.append("- Không có bất thường đáng kể trong cửa sổ quan sát.")
    lines.append("")
    lines.append("## Khuyến nghị chuyên khoa")
    if result.recommendations:
        for s in result.recommendations:
            lines.append(f"- {s}")
    else:
        lines.append("- Duy trì theo dõi định kỳ; chưa cần chuyển chuyên khoa.")
    lines.append("")
    lines.append(f"> *{result.disclaimer}*")
    return "\n".join(lines)


def render_json(patient_id: str, result: RiskResult) -> dict:
    """Trả về JSON theo schema 3 thành phần bắt buộc."""
    return {"patient_id": patient_id, **result.to_dict()}


def save_report(patient_id: str, result: RiskResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{patient_id}.md"
    path.write_text(render_markdown(patient_id, result), encoding="utf-8")
    return path
