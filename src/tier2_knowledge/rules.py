"""Rule engine Tầng 2 — đối chiếu bất thường với cơ sở tri thức y khoa."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.tier1_anomaly import AnomalyRecord

DEFAULT_KB = Path(__file__).parent / "knowledge_base.json"

# Bản đồ fallback: chỉ số → (hệ cơ quan, chuyên khoa) dùng khi chỉ số bất thường
# (Tầng 1 flag) nhưng chưa vượt ngưỡng luật cứng ở Tầng 2.
METRIC_SYSTEM_MAP: dict[str, tuple[str, str]] = {
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

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "system": self.system,
            "system_label": self.system_label,
            "severity": self.severity,
            "specialty": self.specialty,
            "evidence": self.evidence,
            "matched_metrics": self.matched_metrics,
        }


class KnowledgeBase:
    """Nạp cơ sở tri thức JSON và đánh giá luật."""

    def __init__(self, path: str | Path = DEFAULT_KB) -> None:
        with open(path, encoding="utf-8") as f:
            self.data: dict[str, Any] = json.load(f)
        self.rules = self.data["rules"]
        self.system_labels = self.data.get("system_labels", {})

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

    def evaluate(self, snapshot: dict[str, float]) -> list[RuleHit]:
        """Đối chiếu snapshot chỉ số hiện tại với mọi luật."""
        hits: list[RuleHit] = []
        for rule in self.rules:
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
                    )
                )
        return sorted(hits, key=lambda h: -h.severity)

    def evaluate_from_records(self, records: list[AnomalyRecord]) -> list[RuleHit]:
        """Phiên bản dùng chính kết quả Tầng 1 (giá trị mới nhất)."""
        snapshot = {r.metric: r.current for r in records}
        return self.evaluate(snapshot)

    def suggest_for_flagged(self, records: list[AnomalyRecord]) -> dict[str, str]:
        """Bản đồ fallback: chỉ số bị flag (Tầng 1) → hệ cơ quan & chuyên khoa.

        Trả về {system_label: specialty} cho các chỉ số bất thường chưa được
        luật cứng nào bắt, để hệ thống vẫn cảnh báo "theo dõi thêm".
        """
        suggestions: dict[str, str] = {}
        for r in records:
            if r.flagged and r.metric in METRIC_SYSTEM_MAP:
                sys_key, specialty = METRIC_SYSTEM_MAP[r.metric]
                label = self.system_labels.get(sys_key, sys_key)
                suggestions[label] = specialty
        return suggestions
