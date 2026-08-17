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


def explain_patient(
    values: dict[str, float],
    model_keys: list[str] | None = None,
    n_features: int = 5,
) -> list[dict[str, Any]]:
    """Chạy từng model trên 1 bệnh nhân + luận giải feature đóng góp.

    Luận giải model-agnostic bằng **perturbation**: thay lần lượt từng đặc
    trưng bằng median quần thể (từ dataset), đo độ giảm điểm nguy cơ -> đặc
    trưng nào đẩy điểm lên cao (dương) / kéo xuống (âm). Không cần SHAP, áp
    được cho mọi model — minh chứng "cùng 1 ca, các model luận giải ra sao".
    """
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
        row = row.reindex(columns=NHANES_FEATURES)
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
        out.append(
            {
                **_model_meta(key),
                "risk_score": round(proba, 4),
                "level": _level(proba),
                "top_features": contributions[:n_features],
            }
        )
    return out


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