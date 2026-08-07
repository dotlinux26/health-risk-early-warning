"""Sinh báo cáo đầu ra dạng văn bản cấu trúc (markdown)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tier3_risk.scoring import RiskResult


def _fmt_value(row: dict[str, Any]) -> str:
    unit = row["unit"]
    return f"{row['current']:.1f} {unit}".strip() if unit else f"{row['current']:.1f}"


def _fmt_delta(row: dict[str, Any]) -> str:
    if row["delta"] is None:
        return "—"
    d, p = row["delta"], row["pct_change"]
    if p is None:
        return f"{d:+.1f}"
    return f"{d:+.1f} ({p:+.1f}%)"


def _fmt_range(row: dict[str, Any]) -> str:
    if row["range_lo"] is None:
        return "—"
    unit = row["unit"]
    return f"{row['range_lo']:.1f}–{row['range_hi']:.1f} {unit}".strip()


def render_markdown(patient_id: str, result: RiskResult) -> str:
    """Xuất báo cáo markdown chuẩn hóa cho một bệnh nhân.

    Thứ tự trình bày cố ý đặt quyết định theo luật và chỉ số TRƯỚC, sau đó mới
    đến hỗ trợ của mô hình ML — vì AI chỉ là nguồn bổ sung, không phải chẩn đoán.
    """
    lines: list[str] = []
    lines.append(f"# Báo cáo đánh giá nguy cơ sức khỏe — Bệnh nhân {patient_id}")
    lines.append("")
    lines.append(f"- **Phân loại rủi ro:** {result.risk_level} (điểm {result.risk_score:.3f})")
    lines.append(f"- **Hệ cơ quan có khả năng ảnh hưởng:** "
                 + (", ".join(result.affected_systems) if result.affected_systems else "Chưa xác định"))
    lines.append("")

    # 1) Luật lâm sàng được kích hoạt (quyết định theo tri thức y khoa).
    rule_ev = [e for e in result.evidence if e.get("rule_id")]
    lines.append("## Cảnh báo theo luật lâm sàng")
    if rule_ev:
        for e in rule_ev:
            url = e.get("source_url")
            src = f" — nguồn: [{e.get('rule_id')}]({url})" if url else ""
            lines.append(f"- **{e['rule']}**: {e['message']}{src}")
    else:
        lines.append("- Không có luật lâm sàng nào kích hoạt trong cửa sổ quan sát.")
    lines.append("")

    # 2) Chỉ số theo dõi (minh chứng dữ liệu thô).
    if result.metrics_detail:
        lines.append("## Chi tiết các chỉ số theo dõi")
        lines.append("")
        lines.append("| Chỉ số | Giá trị | Đường cơ sở | Thay đổi | Xu hướng | Z-Score | Phạm vi bình thường | Trạng thái |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in result.metrics_detail:
            z = f"{row['z_score']:+.2f}" if row["z_score"] is not None else "—"
            trend = row["trend_name"] or "—"
            base = f"{row['baseline_mean']:.1f}" if row["baseline_mean"] is not None else "—"
            flag = " ⚠" if row["flagged"] else ""
            lines.append(
                f"| {row['name']} | {_fmt_value(row)} | {base} | {_fmt_delta(row)} | "
                f"{trend} | {z} | {_fmt_range(row)} | {row['range_status']}{flag} |"
            )
        lines.append("")
        notes = []
        for row in result.metrics_detail:
            if row["range_status"] not in ("CAO", "THẤP"):
                continue
            name = row["name"]
            unit = f" {row['unit']}" if row["unit"] else ""
            if row["deviation"] is not None:
                notes.append(
                    f"- **{name}** đạt {row['current']:.1f}{unit}, vượt giới hạn "
                    f"{'trên' if row['range_status'] == 'CAO' else 'dưới'} "
                    f"{row['deviation']:.1f}{unit}."
                )
            else:
                notes.append(
                    f"- **{name}** đạt {row['current']:.1f}{unit}, ngoài phạm vi bình thường."
                )
            if row["baseline_mean"] is not None and row["delta"] is not None:
                notes.append(
                    f"  So với đường cơ sở cá nhân {row['baseline_mean']:.1f}{unit}: "
                    f"thay đổi {row['delta']:+.1f}{unit} "
                    f"({row['pct_change']:+.1f}%), xu hướng {row['trend_name']}."
                )
        if notes:
            lines.append("### Đánh giá chi tiết từng chỉ số")
            lines.extend(notes)
            lines.append("")

    # 3) Minh chứng thống kê cá nhân hóa (Tầng 1) còn lại.
    stat_ev = [e for e in result.evidence if not e.get("rule_id")]
    lines.append("## Giải thích thông số (minh chứng dữ liệu)")
    if stat_ev:
        for e in stat_ev:
            lines.append(f"- {e['message']}")
    else:
        lines.append("- Không có bất thường đáng kể trong cửa sổ quan sát.")
    lines.append("")

    # 4) Hỗ trợ của mô hình ML — nêu rõ đây chỉ là suy luận bổ sung.
    lines.append("## Hỗ trợ của mô hình học máy (bổ sung)")
    if result.components.get("ml", 0.0) > 0:
        lines.append(
            f"- Điểm nguy cơ mô hình ML: **{result.components['ml']:.2f}** — "
            f"{'có xu hướng cảnh báo' if result.components['ml'] >= 0.5 else 'thấp'}."
        )
    else:
        lines.append("- Mô hình ML chưa đóng góp (chưa đủ dữ liệu hoặc chưa kích hoạt).")
    lines.append(
        "- Đây là **suy luận bổ sung dựa trên tình trạng bệnh lý**, không phải "
        "chẩn đoán chính thức; kết luận cuối phải do bác sĩ xác nhận."
    )
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
