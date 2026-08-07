"""Giải thích mô hình bằng SHAP (điểm mạnh của LightGBM tại Tầng 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.risk_model import RiskModel


def global_importance(model: RiskModel, X: pd.DataFrame) -> pd.DataFrame:
    """Tầm quan trọng đặc trưng toàn cục (feature importance chuẩn hóa)."""
    importances = model.model.feature_importances_
    if importances.sum() == 0:
        return pd.DataFrame({"feature": X.columns, "importance": 0.0})
    return (
        pd.DataFrame({"feature": X.columns, "importance": importances / importances.sum()})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def local_explanation(model: RiskModel, X: pd.DataFrame) -> list[dict]:
    """SHAP local cho từng bệnh nhân — lý do cụ thể của từng đặc trưng.

    Yêu cầu thư viện shap (pip install shap).
    """
    try:
        import shap
    except ImportError:
        return [{"error": "Chưa cài shap"}]

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    rows: list[dict] = []
    for i in range(len(X)):
        contrib = dict(zip(X.columns, shap_values[i]))
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:10]
        rows.append(
            {
                "patient": X.index[i] if X.index.name else i,
                "top_contributors": [
                    {"feature": f, "shap_value": round(v, 4)} for f, v in top
                ],
            }
        )
    return rows
