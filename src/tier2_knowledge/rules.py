"""Rule engine Tầng 2 — đối chiếu bất thường với cơ sở tri thức y khoa.

Cơ sở tri thức là file JSON **mở rộng được** (không khóa cứng): bác sĩ có thể
thêm hệ cơ quan mới (`system_labels`), chỉ số mới (`metrics`) và luật mới
(`rules`) qua API / giao diện. Tất cả thay đổi được validate trước khi ghi.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tier1_anomaly import AnomalyRecord

DEFAULT_KB = Path(__file__).parent / "knowledge_base.json"

# Các toán tử so sánh hợp lệ trong điều kiện luật.
SUPPORTED_OPERATORS: set[str] = {">", ">=", "<", "<=", "=="}

# Định dạng mã luật gợi ý: R_<HỆ>_<SỐ> (không bắt buộc nhưng khuyến nghị).
RULE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{2,32}$")

# Bản đồ fallback mặc định: chỉ số → (hệ cơ quan, chuyên khoa) khi luật chưa
# khai hệ. Chỉ dùng khi chỉ số chưa được khai báo trong `metrics` của KB.
_DEFAULT_METRIC_SYSTEM: dict[str, tuple[str, str]] = {
    "systolic_bp": ("tim_manh", "Khoa Tim mạch"),
    "diastolic_bp": ("tim_manh", "Khoa Tim mạch"),
    "heart_rate": ("tim_manh", "Khoa Tim mạch"),
    "glucose": ("noi_tiet", "Khoa Nội tiết"),
    "glucose_fasting": ("noi_tiet", "Khoa Nội tiết"),
    "hba1c": ("noi_tiet", "Khoa Nội tiết"),
    "creatinine": ("than", "Khoa Thận"),
    "egfr": ("than", "Khoa Thận"),
    "spo2": ("ho_hap", "Khoa Hô hấp"),
    "bmi": ("chuyen_hoa", "Khoa Nội tiết"),
}

# Chế độ chẩn đoán chuyên biệt: mode -> nhóm hệ cơ quan được xét.
# Ví dụ mode="htn" chỉ xét các luật của hệ tim mạch (tăng huyết áp).
DIAGNOSTIC_MODES: dict[str, set[str]] = {
    "all": set(),                            # xét mọi luật (không lọc)
    "htn": {"tim_manh"},                     # tăng huyết áp
    "dm": {"noi_tiet"},                      # đái tháo đường
    "ckd": {"than"},                         # suy thận mạn
    "resp": {"ho_hap"},                      # hô hấp / SpO2
    "met": {"chuyen_hoa"},                   # chuyển hóa / cân nặng
    "cv": {"tim_manh"},                      # tim mạch tổng hợp
    "endo": {"noi_tiet", "chuyen_hoa"},      # nội tiết + chuyển hóa
}


def normalize_modes(modes: list[str] | None) -> list[str] | None:
    """Chuẩn hoá danh sách mode; None/"all" -> None (xét tất cả)."""
    if not modes:
        return None
    out = {m.strip().lower() for m in modes if m and m.strip().lower() != "all"}
    return list(out) if out else None


def _rule_in_modes(rule: dict[str, Any], modes: list[str]) -> bool:
    system = rule.get("system", "")
    allowed: set[str] = set()
    for m in modes:
        allowed |= DIAGNOSTIC_MODES.get(m, {m})
    return system in allowed


def validate_condition(condition: Any, errors: list[str], prefix: str = "condition") -> None:
    """Validate cấu trúc điều kiện luật (đệ quy, chấp nhận and/or lồng nhau).

    Chỉ số (metric) KHÔNG bị giới hạn bởi danh sách cố định — cho phép mở rộng;
    chỉ kiểm tra cấu trúc và kiểu dữ liệu để tránh lỗi chạy thời điểm thực thi.
    """
    if not isinstance(condition, dict):
        errors.append(f"{prefix}: phải là object điều kiện")
        return
    if "logic" in condition:
        logic = condition.get("logic")
        if logic not in {"and", "or"}:
            errors.append(f"{prefix}.logic: chỉ nhận 'and' hoặc 'or'")
        conds = condition.get("conditions")
        if not isinstance(conds, list) or not conds:
            errors.append(f"{prefix}.conditions: cần ít nhất một điều kiện con")
            return
        for i, c in enumerate(conds):
            validate_condition(c, errors, f"{prefix}.conditions[{i}]")
        return
    metric = condition.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        errors.append(f"{prefix}.metric: cần khai báo tên chỉ số")
    op = condition.get("op")
    if op not in SUPPORTED_OPERATORS:
        errors.append(f"{prefix}.op: toán tử phải thuộc {sorted(SUPPORTED_OPERATORS)}")
    threshold = condition.get("threshold")
    if not isinstance(threshold, (int, float)):
        errors.append(f"{prefix}.threshold: cần là số")
    for extra in ("window_days", "unit"):
        if extra in condition and not isinstance(condition[extra], (str, int, float)):
            errors.append(f"{prefix}.{extra}: giá trị không hợp lệ")


def validate_rule(rule: dict[str, Any]) -> list[str]:
    """Validate một luật trước khi lưu. Trả về danh sách lỗi (rỗng = hợp lệ)."""
    errors: list[str] = []
    rule_id = rule.get("rule_id")
    if not isinstance(rule_id, str) or not RULE_ID_RE.match(rule_id):
        errors.append("rule_id: cần chuỗi 2-32 ký tự (chữ, số, _ , -)")
    if not isinstance(rule.get("name"), str) or not rule["name"].strip():
        errors.append("name: không được để trống")
    if not isinstance(rule.get("system"), str) or not rule["system"].strip():
        errors.append("system: cần khai báo hệ cơ quan")
    severity = rule.get("severity")
    if not isinstance(severity, (int, float)) or not (0.0 <= severity <= 1.0):
        errors.append("severity: cần số trong khoảng 0.0-1.0")
    if not isinstance(rule.get("specialty"), str) or not rule["specialty"].strip():
        errors.append("specialty: không được để trống")
    if not isinstance(rule.get("evidence"), str) or not rule["evidence"].strip():
        errors.append("evidence: cần ghi nguồn trích dẫn")
    source_url = rule.get("source_url")
    if source_url and not (isinstance(source_url, str) and source_url.startswith(("http://", "https://"))):
        errors.append("source_url: cần link http/https hợp lệ")
    # Metadata nguồn chi tiết — tùy chọn, có thì hiện thêm, không thì bỏ qua.
    for field in ("source_page", "source_section", "source_excerpt"):
        val = rule.get(field)
        if val is not None and not isinstance(val, str):
            errors.append(f"{field}: cần là chuỗi văn bản")
    # Chế độ chẩn đoán chuyên biệt — tùy chọn. Mặc định "all" nếu không khai.
    modes = rule.get("modes")
    if modes is not None:
        if not isinstance(modes, list) or not all(isinstance(m, str) for m in modes):
            errors.append("modes: cần là danh sách chuỗi (vd [\"htn\", \"cv\"])")
        else:
            from src.tier2_knowledge.rules import DIAGNOSTIC_MODES

            valid = set(DIAGNOSTIC_MODES) | {"all"}
            for m in modes:
                if m not in valid:
                    errors.append(f"modes: '{m}' không hợp lệ (các mode: {sorted(valid)})")
    validate_condition(rule.get("condition"), errors)
    return errors


@dataclass
class RuleHit:
    """Một luật lâm sàng được kích hoạt."""

    rule_id: str
    name: str
    system: str
    system_label: str
    severity: float
    specialty: str
    evidence: str
    matched_metrics: list[str] = field(default_factory=list)
    modes: list[str] = field(default_factory=list)
    source_url: str = ""
    source_page: str = ""
    source_section: str = ""
    source_excerpt: str = ""

    def to_dict(self) -> dict:
        d = {
            "rule_id": self.rule_id,
            "name": self.name,
            "system": self.system,
            "system_label": self.system_label,
            "severity": self.severity,
            "specialty": self.specialty,
            "evidence": self.evidence,
            "matched_metrics": self.matched_metrics,
            "modes": self.modes,
            "source_url": self.source_url,
        }
        if self.source_page:
            d["source_page"] = self.source_page
        if self.source_section:
            d["source_section"] = self.source_section
        if self.source_excerpt:
            d["source_excerpt"] = self.source_excerpt
        return d


class KnowledgeBase:
    """Nạp cơ sở tri thức JSON, đánh giá luật, và cho phép cập nhật (CRUD).

    Cấu trúc file:
        meta            metadata (version, nguồn)
        metrics         metadata chỉ số: metric -> {name, unit, range}
        system_labels   bản đồ mã hệ -> tên hệ (mở rộng được)
        rules           danh sách luật
    """

    def __init__(self, path: str | Path = DEFAULT_KB) -> None:
        self.path = Path(path)
        self.reload()

    def reload(self) -> None:
        """(Đ)ọc lại file — dùng sau khi API ghi đè để scorer/agent nhận luật mới."""
        with open(self.path, encoding="utf-8") as f:
            self.data: dict[str, Any] = json.load(f)
        from src.tier2_knowledge.governance import ensure_governance

        migrated = False
        for r in self.data["rules"]:
            if "status" not in r:
                ensure_governance(r)
                migrated = True
        self.rules: list[dict[str, Any]] = self.data["rules"]
        self.system_labels: dict[str, str] = self.data.get("system_labels", {})
        self.metrics: dict[str, dict[str, Any]] = self.data.get("metrics", {})
        if migrated:
            self.save()

    # ------------------------------------------------------------------ #
    # Trợ giúp metadata chỉ số (mở rộng)
    # ------------------------------------------------------------------ #
    def metric_info(self, metric: str) -> dict[str, Any]:
        """Trả metadata của một chỉ số; fallback về bản đồ mặc định."""
        info = self.metrics.get(metric) or {}
        if metric in _DEFAULT_METRIC_SYSTEM:
            sys_key, specialty = _DEFAULT_METRIC_SYSTEM[metric]
            info.setdefault("system", sys_key)
            info.setdefault("specialty", specialty)
        return info

    def metric_system_fallback(self, metric: str) -> tuple[str, str] | None:
        """(hệ_key, chuyên_khoa) fallback khi chỉ số bị flag chưa có luật cứng."""
        info = self.metrics.get(metric)
        if info and info.get("system") and info.get("specialty"):
            return info["system"], info["specialty"]
        return _DEFAULT_METRIC_SYSTEM.get(metric)

    # ------------------------------------------------------------------ #
    # Đánh giá luật
    # ------------------------------------------------------------------ #
    def _compare(self, value: float, op: str, threshold: float) -> bool:
        if value is None:
            return False
        return {
            ">": value > threshold,
            ">=": value >= threshold,
            "<": value < threshold,
            "<=": value <= threshold,
            "==": value == threshold,
        }[op]

    def _eval_condition(self, condition: dict, snapshot: dict[str, float]) -> bool:
        """Đánh giá một điều kiện (có thể lồng and/or)."""
        if "logic" in condition:
            results = [self._eval_condition(c, snapshot) for c in condition["conditions"]]
            return all(results) if condition["logic"] == "and" else any(results)
        return self._compare(snapshot.get(condition["metric"]), condition["op"], condition["threshold"])

    def _collect_metrics(self, condition: dict, out: list[str]) -> None:
        if "logic" in condition:
            for c in condition["conditions"]:
                self._collect_metrics(c, out)
        else:
            out.append(condition["metric"])

    def evaluate(
        self,
        snapshot: dict[str, float],
        modes: list[str] | None = None,
        include_inactive: bool = False,
    ) -> list[RuleHit]:
        """Đối chiếu snapshot chỉ số hiện tại với mọi luật.

        modes: lọc luật theo chế độ chẩn đoán chuyên biệt. Mỗi luật khai báo
        thuộc hệ cơ quan nào (field `system`); mode ánh xạ system -> nhóm bệnh.
        Ví dụ mode=["htn"] chỉ xét luật hệ tim mạch (tăng huyết áp).

        Governance (P1): mặc định CHỈ luật ở trạng thái 'active' tham gia chấm
        điểm production. include_inactive=True để preview toàn bộ (UI quản trị).
        """
        hits: list[RuleHit] = []
        for rule in self.rules:
            if not include_inactive and rule.get("status", "active") != "active":
                continue
            if modes and not _rule_in_modes(rule, modes):
                continue
            cond = rule["condition"]
            if self._eval_condition(cond, snapshot):
                metrics: list[str] = []
                self._collect_metrics(cond, metrics)
                hits.append(
                    RuleHit(
                        rule_id=rule["rule_id"],
                        name=rule["name"],
                        system=rule["system"],
                        system_label=self.system_labels.get(rule["system"], rule["system"]),
                        severity=rule["severity"],
                        specialty=rule["specialty"],
                        evidence=rule["evidence"],
                        matched_metrics=metrics,
                        modes=rule.get("modes", ["all"]),
                        source_url=rule.get("source_url", ""),
                        source_page=rule.get("source_page", ""),
                        source_section=rule.get("source_section", ""),
                        source_excerpt=rule.get("source_excerpt", ""),
                    )
                )
        return sorted(hits, key=lambda h: -h.severity)

    def evaluate_from_records(self, records: list[AnomalyRecord], modes: list[str] | None = None) -> list[RuleHit]:
        """Phiên bản dùng chính kết quả Tầng 1 (giá trị mới nhất)."""
        snapshot = {r.metric: r.current for r in records}
        return self.evaluate(snapshot, modes=modes)

    def suggest_for_flagged(self, records: list[AnomalyRecord]) -> dict[str, str]:
        """Bản đồ fallback: chỉ số bị flag (Tầng 1) → hệ cơ quan & chuyên khoa.

        Trả về {system_label: specialty} cho các chỉ số bất thường chưa được
        luật cứng nào bắt, để hệ thống vẫn cảnh báo "theo dõi thêm".
        """
        suggestions: dict[str, str] = {}
        for r in records:
            if not r.flagged:
                continue
            fallback = self.metric_system_fallback(r.metric)
            if fallback:
                sys_key, specialty = fallback
                label = self.system_labels.get(sys_key, sys_key)
                suggestions[label] = specialty
        return suggestions

    # ------------------------------------------------------------------ #
    # Cập nhật (CRUD) — kháng lỗi, ghi an toàn
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """Ghi file JSON an toàn: validate cấu trúc, backup bản cũ trước khi ghi."""
        if not isinstance(self.data, dict) or not isinstance(self.rules, list):
            raise ValueError("Dữ liệu cơ sở tri thức không hợp lệ — từ chối ghi.")
        self.data["rules"] = self.rules
        self.data["system_labels"] = self.system_labels
        self.data["metrics"] = self.metrics
        if self.path.exists():
            backup = self.path.with_suffix(".json.bak")
            shutil.copy2(self.path, backup)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        tmp.replace(self.path)
        self.reload()

    def add_rule(self, rule: dict[str, Any], actor: str = "unknown") -> dict[str, Any]:
        """Thêm luật mới (mặc định DRAFT v1.0 — phải qua duyệt mới chạy production)."""
        errors = validate_rule(rule)
        if errors:
            raise ValueError("; ".join(errors))
        if any(r["rule_id"] == rule["rule_id"] for r in self.rules):
            raise ValueError(f"rule_id '{rule['rule_id']}' đã tồn tại.")
        clean = {k: rule[k] for k in (
            "rule_id", "name", "system", "condition", "severity",
            "specialty", "evidence", "source_url",
        )}
        for k in ("source_page", "source_section", "source_excerpt"):
            v = rule.get(k)
            if isinstance(v, str) and v.strip():
                clean[k] = v.strip()
        from src.tier2_knowledge.governance import ensure_governance, log_audit, now_iso

        # Luật mới luôn bắt đầu ở draft — an toàn lâm sàng mặc định.
        ensure_governance(clean)
        clean["status"] = "draft"
        clean["rule_version"] = 1.0
        clean["created_by"] = actor
        clean["updated_at"] = now_iso()
        self.rules.append(clean)
        self.save()
        log_audit(actor, "create", clean["rule_id"], detail={"version": 1.0})
        return clean

    def update_rule(self, rule_id: str, patch: dict[str, Any],
                    actor: str = "unknown") -> dict[str, Any]:
        """Cập nhật nội dung luật -> tự động bump phiên bản và reset về draft."""
        from src.tier2_knowledge.governance import bump_version, log_audit

        for i, rule in enumerate(self.rules):
            if rule["rule_id"] != rule_id:
                continue
            old_version = float(rule.get("rule_version", 1.0))
            merged = dict(rule)
            for k in ("name", "system", "condition", "severity",
                      "specialty", "evidence", "source_url"):
                if k in patch:
                    merged[k] = patch[k]
            for k in ("source_page", "source_section", "source_excerpt"):
                if k in patch:
                    v = patch[k]
                    if isinstance(v, str) and v.strip():
                        merged[k] = v.strip()
                    else:
                        merged.pop(k, None)
            errors = validate_rule(merged)
            if errors:
                raise ValueError("; ".join(errors))
            changed_content = any(
                k in patch and patch[k] != rule.get(k)
                for k in ("name", "system", "condition", "severity",
                          "specialty", "evidence", "source_url")
            )
            if changed_content:
                merged = bump_version(merged, actor)
                self.rules[i] = merged
                self.save()
                log_audit(actor, "edit", rule_id, detail={
                    "from_version": old_version,
                    "to_version": merged["rule_version"],
                    "status_reset_to": "draft",
                })
            else:
                self.rules[i] = merged
                self.save()
            return merged
        raise ValueError(f"Không tìm thấy luật '{rule_id}'.")

    def transition(self, rule_id: str, to_status: str, actor: str = "unknown",
                   note: str = "") -> dict[str, Any]:
        """Chuyển trạng thái governance: draft → review → approved → active."""
        from src.tier2_knowledge.governance import apply_transition

        for i, rule in enumerate(self.rules):
            if rule["rule_id"] != rule_id:
                continue
            updated = apply_transition(dict(rule), to_status, actor, note)
            self.rules[i] = updated
            self.save()
            return updated
        raise ValueError(f"Không tìm thấy luật '{rule_id}'.")

    def delete_rule(self, rule_id: str, actor: str = "unknown") -> None:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r["rule_id"] != rule_id]
        if len(self.rules) == before:
            raise ValueError(f"Không tìm thấy luật '{rule_id}'.")
        from src.tier2_knowledge.governance import log_audit

        self.save()
        log_audit(actor, "delete", rule_id)

    def add_system(self, system_key: str, label: str) -> dict[str, str]:
        """Thêm hệ cơ quan mới: system_key -> label."""
        system_key = (system_key or "").strip()
        label = (label or "").strip()
        if not system_key or not label:
            raise ValueError("system_key và label không được để trống.")
        if system_key in self.system_labels:
            raise ValueError(f"Hệ '{system_key}' đã tồn tại.")
        self.system_labels[system_key] = label
        self.save()
        return {system_key: label}

    def add_metric(self, metric: str, info: dict[str, Any]) -> dict[str, Any]:
        """Thêm chỉ số mới kèm metadata (name, unit, range, system, specialty)."""
        metric = (metric or "").strip()
        if not metric:
            raise ValueError("Tên chỉ số không được để trống.")
        info = {
            "name": (info.get("name") or metric).strip(),
            "unit": (info.get("unit") or "").strip(),
            "range": info.get("range"),
            "system": (info.get("system") or "").strip() or None,
            "specialty": (info.get("specialty") or "").strip() or None,
        }
        if info["range"] is not None:
            if (not isinstance(info["range"], (list, tuple))
                    or len(info["range"]) != 2
                    or not all(isinstance(v, (int, float)) for v in info["range"])):
                raise ValueError("range phải là cặp số [min, max].")
            info["range"] = [float(info["range"][0]), float(info["range"][1])]
        self.metrics[metric] = info
        self.save()
        return self.metrics[metric]

    def to_dict(self) -> dict[str, Any]:
        """Toàn bộ KB dạng dict (cho API /api/kb)."""
        return self.data
