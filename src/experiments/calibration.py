"""Hiệu chỉnh xác suất (P0.1 — docs/15 mục 5.1).

Ba phương pháp: none / platt / isotonic. Calibrator CHỈ được fit trên
validation set; test set chỉ dùng để đánh giá. Việc chọn phương pháp tốt nhất
cũng dựa trên validation (Brier trên val), không nhìn test.

Đầu ra của ML thô KHÔNG được gọi là "confidence" hay "xác suất bệnh" khi chưa
hiệu chỉnh — đây là ranh giới diễn giải bắt buộc trong UI và báo cáo.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CALIBRATION_METHODS: tuple[str, ...] = ("none", "platt", "isotonic")


def fit_calibration(
    method: str, proba_val: np.ndarray, y_val: np.ndarray
):
    """Fit calibrator trên validation. Trả về object có .predict(proba)."""
    proba_val = np.asarray(proba_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int)

    if method == "none":
        return _IdentityCalibrator()
    if method == "platt":
        # Platt scaling: hồi quy logistic trên đầu ra thô của model.
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(proba_val.reshape(-1, 1), y_val)
        return lr
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(proba_val, y_val)
        return iso
    raise ValueError(f"Phương pháp hiệu chỉnh không hỗ trợ: {method}")


def apply_calibration(calibrator, proba: np.ndarray) -> np.ndarray:
    """Áp calibrator lên một mảng xác suất."""
    proba = np.asarray(proba, dtype=float)
    if isinstance(calibrator, (_IdentityCalibrator,)):
        return proba.copy()
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
    return np.asarray(calibrator.predict(proba), dtype=float)


class _IdentityCalibrator:
    """Trả nguyên xác suất — dùng cho method 'none'."""

    def predict(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=float)


def expected_calibration_error(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> float:
    """ECE = tổng w_i * |accuracy_i - confidence_i| theo bin đều [0,1]."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-9, 1 - 1e-9)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(proba, bins[1:-1], right=False)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(proba[mask].mean())
        acc = float(y_true[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def compare_methods(
    proba_val: np.ndarray,
    y_val: np.ndarray,
    proba_te: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    """Fit cả 3 phương pháp trên val, đánh giá trên test.

    Chọn phương pháp tốt nhất THEO VALIDATION (brier_val) — không nhìn test
    khi chọn. Kết quả test cho cả 3 phương pháp đều được xuất ra để minh bạch.
    """
    from sklearn.metrics import brier_score_loss

    out: dict = {"methods": {}}
    best_method, best_val_brier = None, float("inf")
    for m in CALIBRATION_METHODS:
        cal = fit_calibration(m, proba_val, y_val)
        p_val = apply_calibration(cal, proba_val)
        p_te = apply_calibration(cal, proba_te)
        brier_val = float(brier_score_loss(y_val, p_val))
        row = {
            "brier_val": brier_val,
            "ece_val": expected_calibration_error(y_val, p_val),
            "brier_test": float(brier_score_loss(y_te, p_te)),
            "ece_test": expected_calibration_error(y_te, p_te),
        }
        if brier_val < best_val_brier:
            best_method, best_val_brier = m, brier_val
        out["methods"][m] = row
        if m != "none":
            out.setdefault("calibrators", {})[m] = cal
    out["selected"] = best_method
    return out
