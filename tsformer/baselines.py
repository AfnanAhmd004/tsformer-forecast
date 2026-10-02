"""Baselines every forecasting model must beat."""
from __future__ import annotations

import numpy as np


class ZeroForecast:
    """Predicts zero return - the random-walk benchmark."""

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.zeros(len(x), dtype=np.float32)


class RidgeAR:
    """Ridge regression on the flattened lookback window."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, x, y):
        X = x.reshape(len(x), -1)
        X = np.hstack([X, np.ones((len(X), 1))])
        reg = self.alpha * np.eye(X.shape[1])
        reg[-1, -1] = 0.0
        self.w_ = np.linalg.solve(X.T @ X + reg, X.T @ y)
        return self

    def predict(self, x):
        X = x.reshape(len(x), -1)
        return (np.hstack([X, np.ones((len(X), 1))]) @ self.w_).astype(np.float32)
