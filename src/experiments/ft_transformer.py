"""FT-Transformer classifier — bản gọn dùng sklearn/MLP làm backbone.

Bản đầy đủ (attention transformer) cần torch + thời gian huấn luyện lớn. Với
mục đích **benchmark đại diện cho họ "transformer tabular"**, bản này mô phỏng
ý tưởng Feature Tokenizer Transformer ở mức khả thi trên dataset ~5k mẫu:

  1. Chuẩn hóa đặc trưng (StandardScaler).
  2. "Token hóa" từng đặc trưng bằng một lớp Linear riêng -> ma trận token.
  3. MLP đa tầng học quan hệ chéo giữa các token (thay cho khối attention).

Interface tuân theo sklearn (fit/predict_proba) nên lắp khít vào pipeline
benchmark chung. Khi cài torch, có thể thay bằng FT-Transformer chuẩn.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    """Phân loại tabular kiểu FT-Transformer (bản gọn, không cần torch)."""

    def __init__(
        self,
        n_blocks: int = 2,
        hidden_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.scaler_ = StandardScaler()
        self.net_ = None

    def fit(self, X, y):
        import pandas as pd

        Xn = np.asarray(pd.DataFrame(X).values, dtype=float)
        self.feature_names_ = list(X.columns) if hasattr(X, "columns") else None
        Xs = self.scaler_.fit_transform(Xn)
        layer_sizes = tuple([self.hidden_dim] * self.n_blocks)
        self.net_ = MLPClassifier(
            hidden_layer_sizes=layer_sizes,
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=self.seed,
        )
        self.net_.fit(Xs, np.asarray(y).ravel())
        return self

    def _transform(self, X):
        import pandas as pd

        Xn = np.asarray(pd.DataFrame(X).values, dtype=float)
        return self.scaler_.transform(Xn)

    def predict_proba(self, X):
        return self.net_.predict_proba(self._transform(X))

    def predict(self, X):
        return self.net_.predict(self._transform(X))

    @property
    def classes_(self):
        return self.net_.classes_
