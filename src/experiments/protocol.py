"""Protocol thực nghiệm — split chuẩn, seeds, bộ metrics thống nhất.

Theo notes/02: khóa cùng dataset, cùng target, cùng information boundary,
cùng train/val/test partitions; mỗi model dùng preprocessing riêng phù hợp.
Test set cuối cùng được khóa lại, không dùng để tuning.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEEDS: list[int] = [42, 52, 62, 72, 82]

SPLIT = (0.70, 0.15, 0.15)  # train / val / test

# Feature dùng chung cho NHANES — khớp schema hệ thống.
NHANES_FEATURES: list[str] = [
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "glucose_fasting",
    "hba1c",
    "creatinine",
    "bmi",
]


@dataclass
class ExperimentConfig:
    """Cấu hình một experiment (EXP-ML-XXX)."""

    model_key: str
    seed: int
    dataset: str
    features: list[str] = field(default_factory=lambda: list(NHANES_FEATURES))
    train_ratio: float = SPLIT[0]
    val_ratio: float = SPLIT[1]
    test_ratio: float = SPLIT[2]

    @property
    def experiment_id(self) -> str:
        return f"EXP-ML-{self.model_key.upper()}-{self.seed:02d}"


def stratify_split(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Chia train/val/test phân tầng theo nhãn (test khóa lại).

    Chia tuần tự: train/test trước (theo tỷ lệ test), sau đó tách val từ train.
    """
    from sklearn.model_selection import train_test_split

    test_frac = config.test_ratio
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    val_frac = config.val_ratio / (1.0 - test_frac)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_frac, stratify=y_tr, random_state=seed
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def evaluate_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Bộ metrics thống nhất cho mọi model."""
    y_pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def aggregate_seeds(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Gộp kết quả nhiều seed -> {metric: {"mean": x, "std": y}}.

    Bỏ qua các key phi số (ví dụ calibration_method là tên phương pháp).
    """
    keys = [
        k for k in rows[0]
        if k not in {"seed"} and isinstance(rows[0][k], (int, float))
    ]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out