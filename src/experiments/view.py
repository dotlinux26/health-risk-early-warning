"""Dựng dữ liệu cho giao diện benchmark (đọc từ experiments/, không train lại).

Cung cấp:
    - build_summary()      -> bảng tổng hợp đa model (mean ± std)
    - list_experiments()   -> danh sách evidence package theo model
    - explain_patient()    -> với 1 bệnh nhân: điểm + luận giải của TỪNG model
                             (model-agnostic: perturbation quanh baseline cá nhân)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.experiments.models import MODEL_SPECS
from src.experiments.protocol import NHANES_FEATURES

EXPERIMENTS_DIR = Path("experiments")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_summary() -> dict[str, Any]:
    """Đọc summary.json (nếu có) + metadata model, trả về cấu trúc cho view."""
    summary = _read_json(EXPERIMENTS_DIR / "summary.json") or {}
    models_out: list[dict[str, Any]] = []
    for key, spec in MODEL_SPECS.items():
        m = summary.get("models", {}).get(key)
        if m is None:
            continue
        row = {
            "key": key,
            "name": spec.name,
            "family": spec.family,
            "description": spec.description,
            "metrics": {k: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)}
                        for k, v in m.items()},
        }
        models_out.append(row)
    models_out.sort(key=lambda r: -r["metrics"].get("roc_auc", {}).get("mean", 0.0))
    return {
        "dataset": summary.get("dataset"),
        "n": summary.get("n"),
        "positive": summary.get("positive"),
        "seeds": summary.get("seeds"),
        "data_completeness": summary.get("data_completeness"),
        "models": models_out,
    }


def list_experiments() -> list[dict[str, Any]]:
    """Danh sách evidence package: model, seed, metrics, có curves.png không."""
    out: list[dict[str, Any]] = []
    if not EXPERIMENTS_DIR.exists():
        return out
    for d in sorted(EXPERIMENTS_DIR.glob("EXP-ML-*")):
        if not d.is_dir():
            continue
        cfg = _read_json(d / "config.json") or {}
        metrics = _read_json(d / "metrics.json") or {}
        imp = d / "feature_importance.csv"
        out.append(
            {
                "experiment_id": d.name,
                "model": cfg.get("model_key"),
                "seed": cfg.get("seed"),
                "metrics": metrics,
                "has_curves": (d / "curves.png").exists(),
                "has_importance": imp.exists(),
                "dir": d.name,
            }
        )
    out.sort(key=lambda e: e["experiment_id"])
    return out


def _model_meta(model_key: str) -> dict[str, Any]:
    spec = MODEL_SPECS.get(model_key)
    return {
        "key": model_key,
        "name": spec.name if spec else model_key,
        "family": spec.family if spec else "",
    }


def _find_model(model_key: str, seed: int = 42):
    """Nạp model.joblib của một experiment (mặc định seed đầu tiên có)."""
    if not EXPERIMENTS_DIR.exists():
        return None
    candidates = sorted(EXPERIMENTS_DIR.glob(f"EXP-ML-{model_key.upper()}-*"))
    for c in candidates:
        m = c / "model.joblib"
        if m.exists():
            return joblib.load(m)
    return None


def available_input_metrics() -> list[dict[str, str]]:
    """Danh sách chỉ số có thể nhập để đánh giá (từ KB + NHANES features)."""
    from src.tier2_knowledge.rules import KnowledgeBase

    labels: dict[str, str] = {}
    try:
        kb = KnowledgeBase()
        for metric, info in kb.metrics.items():
            labels[metric] = info.get("name") or metric
    except Exception:
        pass
    known = [
        "systolic_bp", "diastolic_bp", "heart_rate", "glucose",
        "glucose_fasting", "hba1c", "creatinine", "egfr", "spo2", "bmi",
    ]
    out: list[dict[str, str]] = []
    for m in known:
        out.append({"key": m, "name": labels.get(m) or m})
    for m in NHANES_FEATURES:
        if m not in known:
            out.append({"key": m, "name": labels.get(m) or m})
    return out


def fusion_score(values: dict[str, float], model_keys: list[str] | None = None) -> float | None:
    """Điểm rủi ro fusion: trung bình trọng số (theo AUC benchmark) xác suất
    nguy cơ của MỌI model đã train. None nếu không có model nào sẵn sàng."""
    if not EXPERIMENTS_DIR.exists():
        return None
    from src.experiments.models import available_models

    keys = model_keys or available_models()
    aucs = _model_aucs()
    total_w = 0.0
    total_p = 0.0
    medians = _baseline_medians()
    for key in keys:
        model = _find_model(key)
        if model is None:
            continue
        row = pd.DataFrame([{f: values.get(f) for f in NHANES_FEATURES}])
        row = row.reindex(columns=NHANES_FEATURES).astype("float64")
        for feat in NHANES_FEATURES:
            if pd.isna(row[feat].iloc[0]):
                med = medians.get(feat)
                if med is not None:
                    row[feat] = med
        try:
            proba = float(model.predict_proba(row)[:, 1][0])
        except Exception:
            continue
        w = aucs.get(key, 0.0)
        if w <= 0:
            continue
        total_w += w
        total_p += w * proba
    if total_w <= 0:
        return None
    return total_p / total_w


def _model_aucs() -> dict[str, float]:
    """AUC trung bình từ experiments/summary.json (dict keyed theo model)."""
    import json as _json

    path = EXPERIMENTS_DIR / "summary.json"
    if not path.exists():
        return {}
    try:
        d = _json.loads(path.read_text())
        models = d.get("models", {})
        return {
            k: float(v["roc_auc"]["mean"])
            for k, v in models.items()
            if isinstance(v, dict) and v.get("roc_auc", {}).get("mean") is not None
        }
    except Exception:
        return {}


def explain_patient(
    values: dict[str, float],
    model_keys: list[str] | None = None,
    n_features: int = 5,
    rules: list[dict[str, Any]] | None = None,
    score_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not EXPERIMENTS_DIR.exists():
        return []
    from src.experiments.models import available_models

    keys = model_keys or available_models()
    medians = _baseline_medians()

    out: list[dict[str, Any]] = []
    for key in keys:
        model = _find_model(key)
        if model is None:
            continue
        row = pd.DataFrame([{f: values.get(f) for f in NHANES_FEATURES}])
        row = row.reindex(columns=NHANES_FEATURES).astype("float64")
        # Impute missing bằng median giống lúc train (SimpleImputer median) —
        # model chưa từng thấy NaN khi huấn luyện, nếu để NaN model sẽ cho
        # cùng xác suất cho mọi input -> delta đóng góp ~ 0 vô nghĩa.
        for feat in NHANES_FEATURES:
            if pd.isna(row[feat].iloc[0]):
                med = medians.get(feat)
                if med is not None:
                    row[feat] = med
        try:
            proba = float(model.predict_proba(row)[:, 1][0])
        except Exception:
            continue
        contributions = []
        for feat in NHANES_FEATURES:
            base_val = values.get(feat)
            if base_val is None or np.isnan(base_val):
                continue
            alt = row.copy()
            alt[feat] = medians.get(feat, base_val)
            try:
                alt_proba = float(model.predict_proba(alt)[:, 1][0])
            except Exception:
                continue
            contributions.append(
                {"feature": feat, "delta": round(proba - alt_proba, 4)}
            )
        contributions.sort(key=lambda c: -abs(c["delta"]))

        # Điểm nguy cơ theo TỪNG LUẬT kích hoạt: đặt đúng các chỉ số mà luật
        # đó dùng về baseline quần thể -> độ giảm điểm = mức đóng góp của luật.
        rule_attrib: list[dict[str, Any]] = []
        for rule in rules or []:
            metrics = rule.get("matched_metrics") or []
            metric_subset = [m for m in metrics if m in NHANES_FEATURES]
            if not metric_subset:
                continue
            alt = row.copy()
            for feat in metric_subset:
                med = medians.get(feat)
                if med is not None:
                    alt[feat] = med
            try:
                alt_proba = float(model.predict_proba(alt)[:, 1][0])
            except Exception:
                continue
            rule_attrib.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "name": rule.get("name"),
                    "system_label": rule.get("system_label"),
                    "severity": rule.get("severity"),
                    "matched_metrics": metric_subset,
                    "delta": round(proba - alt_proba, 4),
                }
            )
        rule_attrib.sort(key=lambda r: -r["delta"])

        entry: dict[str, Any] = {
            **_model_meta(key),
            "risk_score": round(proba, 4),
            "level": _level(proba),
            "top_features": contributions[:n_features],
            "rule_attribution": rule_attrib,
        }
        # Điểm tổng hợp 3 tầng riêng của model này (stat + knowledge + model
        # + trend) — để so sánh mức khác biệt giữa các model.
        if score_context:
            comp = dict(score_context.get("components") or {})
            comp["ml"] = min(1.0, proba)
            w = score_context.get("weights") or {}
            total = sum(comp.get(k, 0.0) * w[k] for k in w)
            total = round(min(1.0, max(0.0, total)), 3)
            if score_context.get("critical"):
                total = max(total, score_context.get("floor", 0.5))
                total = round(total, 3)
            entry["total_score"] = total
            entry["total_level"] = _level(total)
        out.append(entry)
    return out


def clinical_context(values: dict[str, float]) -> dict[str, Any]:
    """Bối cảnh lâm sàng của ca: hệ/bệnh nào bị nghi ngờ + luật nào đã kích hoạt.

    Tái sử dụng đúng rule engine của hệ thống (Tầng 2) — đối chiếu snapshot
    chỉ số với knowledge_base.json, trả về các luật thoả. Trả lời: "nguy cơ
    cao của bệnh gì, dựa trên luật nào".
    """
    from src.tier2_knowledge.rules import KnowledgeBase

    snapshot = {k: v for k, v in values.items() if v is not None}
    if not snapshot:
        return {"systems": [], "rules": [], "note": "Không có chỉ số để đối chiếu luật."}
    try:
        kb = KnowledgeBase()
        hits = kb.evaluate(snapshot)
    except Exception as e:
        return {"systems": [], "rules": [], "note": f"Không đối chiếu được luật: {e}"}
    if not hits:
        return {
            "systems": [],
            "rules": [],
            "note": "Không có luật lâm sàng nào kích hoạt với bộ chỉ số này.",
        }

    systems: dict[str, list[dict[str, Any]]] = {}
    rules_out: list[dict[str, Any]] = []
    for h in hits:
        rules_out.append(
            {
                "rule_id": h.rule_id,
                "name": h.name,
                "system": h.system,
                "system_label": h.system_label,
                "severity": h.severity,
                "specialty": h.specialty,
                "matched_metrics": h.matched_metrics,
                "evidence": h.evidence,
            }
        )
        systems.setdefault(h.system_label, []).append(
            {"rule_id": h.rule_id, "name": h.name, "severity": h.severity}
        )
    system_summary = [
        {"label": label, "max_severity": max(r["severity"] for r in rules), "rules": rules}
        for label, rules in systems.items()
    ]
    system_summary.sort(key=lambda s: -s["max_severity"])
    return {"systems": system_summary, "rules": rules_out, "note": ""}


def _level(score: float) -> str:
    if score >= 0.66:
        return "CAO"
    if score >= 0.33:
        return "TRUNG_BINH"
    return "THAP"


def _baseline_medians() -> dict[str, float]:
    """Median quần thể từ NHANES để làm baseline khi perturb từng feature."""
    path = Path("data/datasets/nhanes_2017_2018.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=lambda c: c in NHANES_FEATURES)
    return {c: float(df[c].median()) for c in NHANES_FEATURES if c in df.columns}