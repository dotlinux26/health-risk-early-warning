"""Quản trị tri thức luật lâm sàng (P1 — docs/15 mục 7, WS4).

Biến /rules từ "trang sửa JSON" thành knowledge governance interface:

    Workflow: DRAFT → REVIEW → APPROVED → ACTIVE (nhánh phụ: REJECTED)
    - Chỉ luật ACTIVE tham gia chấm điểm production.
    - Mọi thay đổi nội dung làm reset về DRAFT và tăng phiên bản.
    - Toàn bộ thao tác ghi audit trail (ai, khi nào, cũ → mới).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_PATH = Path("data/kb/audit_log.jsonl")

STATUSES = ("draft", "review", "approved", "active", "rejected")

# Chuyển trạng thái hợp lệ: (từ) -> {được phép đến}
TRANSITIONS: dict[str, set[str]] = {
    "draft": {"review", "rejected"},
    "review": {"approved", "rejected", "draft"},
    "approved": {"active", "review", "draft"},
    "active": {"draft"},   # sửa luật đang chạy -> phải qua vòng duyệt mới
    "rejected": {"draft", "review"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_governance(rule: dict[str, Any]) -> dict[str, Any]:
    """Bổ sung trường governance cho luật cũ (migration khi load KB)."""
    rule.setdefault("status", "active")          # luật hiện hữu = đang chạy
    rule.setdefault("rule_version", 1.0)
    rule.setdefault("created_by", "legacy")
    rule.setdefault("approved_by", None)
    rule.setdefault("approved_at", None)
    rule.setdefault("updated_at", None)
    rule.setdefault("previous_version", None)
    return rule


def bump_version(rule: dict[str, Any], actor: str) -> dict[str, Any]:
    """Sửa nội dung luật -> tăng phiên bản nhỏ, reset về draft."""
    old = float(rule.get("rule_version", 1.0))
    rule["previous_version"] = old
    rule["rule_version"] = round(old + 0.1, 1)
    rule["status"] = "draft"
    rule["updated_at"] = now_iso()
    rule["updated_by"] = actor
    return rule


def validate_transition(rule: dict[str, Any], to_status: str) -> None:
    """Kiểm tra chuyển trạng thái hợp lệ; ném ValueError nếu vi phạm."""
    to_status = to_status.strip().lower()
    if to_status not in STATUSES:
        raise ValueError(f"Trạng thái không hợp lệ: '{to_status}'. "
                         f"Cho phép: {', '.join(STATUSES)}")
    cur = str(rule.get("status", "draft"))
    if to_status == cur:
        raise ValueError(f"Luật đã ở trạng thái '{cur}'.")
    if to_status not in TRANSITIONS.get(cur, set()):
        raise ValueError(
            f"Không được chuyển {cur} → {to_status}. "
            f"Đi đúng luồng: draft → review → approved → active.")
    if to_status == "active":
        # Luật phải được duyệt (approved) mới chạy production — an toàn lâm sàng.
        pass


def apply_transition(rule: dict[str, Any], to_status: str, actor: str,
                     note: str = "") -> dict[str, Any]:
    validate_transition(rule, to_status)
    rule["previous_status"] = rule.get("status")
    rule["status"] = to_status
    rule["updated_at"] = now_iso()
    rule["updated_by"] = actor
    if to_status == "active":
        rule["approved_at"] = now_iso()
        rule["approved_by"] = actor
    log_audit(actor, "transition", rule["rule_id"], detail={
        "to": to_status, "version": rule.get("rule_version"), "note": note,
    })
    return rule


def log_audit(actor: str, action: str, rule_id: str,
              detail: dict[str, Any] | None = None) -> None:
    """Ghi một dòng audit trail (JSONL append-only)."""
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": now_iso(),
        "actor": actor or "unknown",
        "action": action,
        "rule_id": rule_id,
        "detail": detail or {},
    }
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_audit(limit: int = 200) -> list[dict[str, Any]]:
    """Đọc audit trail (mới nhất trước)."""
    if not AUDIT_PATH.exists():
        return []
    rows = [json.loads(l) for l in AUDIT_PATH.read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return list(reversed(rows[-limit:]))
