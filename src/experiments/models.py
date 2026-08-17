"""Đăng ký các mô hình tham gia benchmark (Core + tùy chọn).

Mỗi mô hình là một factory: `build(seed) -> estimator` với interface sklearn
(`fit`, `predict_proba`). Các mô hình cần thư viện chưa cài (xgboost, torch...)
được đăng ký có điều kiện — chạy được thì đưa vào, không thì bỏ qua, giúp
pipeline thực nghiệm không phụ thuộc toàn bộ môi trường.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from lightgbm import LGBMClassifier

try:
    from xgboost import XGBClassifier

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


@dataclass(frozen=True)
class ModelSpec:
    """Thông tin một mô hình trong registry."""

    key: str
    name: str
    family: str
    available: bool
    description: str
    build: callable = field(repr=False, default=None)


def _spec(
    key: str,
    name: str,
    family: str,
    available: bool,
    description: str,
    build: callable | None = None,
) -> ModelSpec:
    return ModelSpec(
        key=key,
        name=name,
        family=family,
        available=available,
        description=description,
        build=build,
    )


def _build_lr(seed: int):
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )


def _build_rf(seed: int):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _build_lgbm(seed: int):
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=16,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=seed,
        verbosity=-1,
    )


def _build_xgb(seed: int):
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=seed,
    )


def _build_mlp(seed: int):
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=seed,
    )


def _build_ft_transformer(seed: int):
    from src.experiments.ft_transformer import FTTransformerClassifier

    return FTTransformerClassifier(
        n_blocks=3,
        hidden_dim=96,
        seed=seed,
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        _spec(
            "lr",
            "Logistic Regression",
            "Baseline tuyến tính",
            True,
            "Baseline tuyến tính — minh bạch tuyệt đối, dùng làm mốc so sánh.",
            _build_lr,
        ),
        _spec(
            "rf",
            "Random Forest",
            "Tree ensemble",
            True,
            "Random forest — baseline ensemble đối chứng.",
            _build_rf,
        ),
        _spec(
            "lgbm",
            "LightGBM",
            "Gradient boosting",
            True,
            "Model chính hiện tại — gradient boosting trên tree.",
            _build_lgbm,
        ),
        _spec(
            "xgb",
            "XGBoost",
            "Gradient boosting",
            _HAS_XGB,
            "XGBoost — boosting đối chứng (cần cài xgboost).",
            _build_xgb,
        ),
        _spec(
            "mlp",
            "MLP",
            "Neural network",
            True,
            "Mạng nơ-ron truyền thẳng — neural baseline.",
            _build_mlp,
        ),
        _spec(
            "fttransformer",
            "FT-Transformer",
            "Transformer tabular",
            True,
            "Feature Tokenizer Transformer (bản gọn, sklearn — không cần torch).",
            _build_ft_transformer,
        ),
    ]
}


def available_models() -> list[str]:
    """Danh sách key các mô hình sẵn sàng chạy trong môi trường hiện tại."""
    return [k for k, s in MODEL_SPECS.items() if s.available]


def build_model(key: str, seed: int):
    """Tạo instance mô hình theo key. Ném KeyError nếu không tồn tại/không sẵn."""
    spec = MODEL_SPECS[key]
    if not spec.available or spec.build is None:
        raise KeyError(f"Model '{key}' chưa sẵn sàng trong môi trường này.")
    return spec.build(seed)
