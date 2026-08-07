"""Tổng hợp điểm rủi ro từ 3 nguồn (thống kê, tri thức, ML, xu hướng)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Config
from src.tier1_anomaly import AnomalyRecord
from src.tier2_knowledge.rules import KnowledgeBase, RuleHit


@dataclass
class RiskResult:
    """Kết quả cuối cùng của Tầng 3."""

    risk_level: str
    risk_score: float
    affected_systems: list[str]
    evidence: list[dict[str, Any]]
    recommendations: list[str]
    components: dict[str, float]
    disclaimer: str = (
        "Hệ thống chỉ HỖ TRỢ QUYẾT ĐỊNH, không thay thế chẩn đoán của bác sĩ."
    )

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "risk_score": round(self.risk_score, 3),
            "affected_systems": self.affected_systems,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "components": self.components,
            "disclaimer": self.disclaimer,
        }


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class RiskScorer:
    """Kết hợp điểm từ Tầng 1, Tầng 2 và mô hình ML (Tầng 3.2)."""

    def __init__(self, config: Config = Config(), kb: KnowledgeBase | None = None) -> None:
        self.config = config
        self.kb = kb or KnowledgeBase()

    def _stat_score(self, records: list[AnomalyRecord]) -> float:
        if not records:
            return 0.0
        z = [abs(r.z_score or 0) for r in records if r.flagged]
        return min(1.0, (max(z, default=0.0) / 4.0))

    def _knowledge_score(self, hits: list[RuleHit]) -> float:
        if not hits:
            return 0.0
        return min(1.0, max(h.severity for h in hits))

    def _trend_score(self, records: list[AnomalyRecord]) -> float:
        if not records:
            return 0.0
        rising = sum(1 for r in records if r.trend == "rising" and r.flagged)
        return min(1.0, rising / max(1, len(records)) * 2)

    def score(
        self,
        records: list[AnomalyRecord],
        ml_score: float | None = None,
        snapshot: dict[str, float] | None = None,
    ) -> RiskResult:
        """Tính điểm tổng hợp và sinh đầu ra chuẩn 3 thành phần."""
        hits = self.kb.evaluate_from_records(records) if records else []
        if snapshot and not records:
            hits = self.kb.evaluate(snapshot)
        suggestions = self.kb.suggest_for_flagged(records) if records else {}

        components = {
            "stat": self._stat_score(records),
            "knowledge": self._knowledge_score(hits),
            "ml": min(1.0, ml_score or 0.0),
            "trend": self._trend_score(records),
        }
        w = self.config.risk_weights
        total = sum(components[k] * w[k] for k in w)
        total = round(min(1.0, max(0.0, total)), 3)

        # An toàn lâm sàng: luật nghiêm trọng kích hoạt -> ít nhất TRUNG_BINH
        if any(h.severity >= self.config.critical_rule_severity for h in hits):
            total = max(total, self.config.critical_rule_floor)
            total = round(total, 3)

        low, high = self.config.risk_level_thresholds
        level = "CAO" if total >= high else ("TRUNG_BINH" if total >= low else "THAP")

        affected = sorted({h.system_label for h in hits} | set(suggestions.keys()))
        recommendations = sorted({h.specialty for h in hits} | set(suggestions.values()))

        evidence: list[dict[str, Any]] = []
        for r in records:
            if r.flagged:
                evidence.append(
                    {
                        "metric": r.metric,
                        "current": r.current,
                        "baseline_mean": r.baseline_mean,
                        "z_score": r.z_score,
                        "message": (
                            f"{r.metric}: giá trị {r.current:.1f} lệch {r.z_score:+.2f}σ "
                            f"so với trung bình cá nhân {r.baseline_mean:.1f} (xu hướng {r.trend})"
                        ),
                    }
                )
        for h in hits:
            evidence.append(
                {
                    "rule_id": h.rule_id,
                    "rule": h.name,
                    "system": h.system_label,
                    "message": f"Kích hoạt luật '{h.name}' ({h.evidence}) → {h.system_label}.",
                }
            )
        for system_label, specialty in suggestions.items():
            evidence.append(
                {
                    "system": system_label,
                    "message": (
                        f"Có chỉ số bất thường cá nhân hóa (Tầng 1) nhưng chưa vượt "
                        f"ngưỡng luật cứng → khuyến nghị theo dõi {specialty}."
                    ),
                }
            )

        return RiskResult(
            risk_level=level,
            risk_score=total,
            affected_systems=affected,
            evidence=evidence,
            recommendations=recommendations,
            components=components,
        )
