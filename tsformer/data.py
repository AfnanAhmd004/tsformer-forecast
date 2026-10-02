"""Synthetic market-like series, supervised windowing and walk-forward splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def synthetic_returns(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Daily returns with weak, regime-dependent autocorrelation and volatility clustering.

    Real returns are close to unpredictable; this series has a small amount of
    learnable structure so models can be compared honestly against baselines.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    vol = np.full(n, 0.01)
    regime_phi = 0.0
    for t in range(2, n):
        if t % 250 == 0:
            regime_phi = rng.choice([-0.15, 0.0, 0.2])
        vol[t] = np.sqrt(1e-6 + 0.08 * r[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
        r[t] = regime_phi * r[t - 1] + 0.05 * np.sin(2 * np.pi * t / 21) * vol[t] + vol[t] * rng.standard_normal()
    return r


def make_features(returns: np.ndarray) -> np.ndarray:
    """Per-step features: return, |return| and a short rolling mean - all causal."""
    ret = returns
    absr = np.abs(returns)
    roll5 = np.convolve(returns, np.ones(5) / 5, mode="full")[: len(returns)]
    return np.stack([ret, absr, roll5], axis=1).astype(np.float32)


def make_windows(features: np.ndarray, target: np.ndarray, lookback: int, horizon: int = 1):
    """X[i] = features[i : i+lookback], y[i] = sum of target over the next `horizon` steps.

    The label window starts strictly after the input window ends.
    """
    xs, ys = [], []
    for i in range(len(features) - lookback - horizon + 1):
        xs.append(features[i : i + lookback])
        ys.append(target[i + lookback : i + lookback + horizon].sum())
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


@dataclass(frozen=True)
class Fold:
    train: slice
    val: slice
    test: slice


def walk_forward_folds(n: int, n_folds: int = 4, train_frac: float = 0.5, val_frac: float = 0.1, gap: int = 0):
    """Expanding-window folds over window indices; `gap` drops samples between sets."""
    start_test = int(n * (train_frac + val_frac))
    test_len = (n - start_test) // n_folds
    folds = []
    for k in range(n_folds):
        test_start = start_test + k * test_len
        val_len = int(n * val_frac)
        val_start = test_start - gap - val_len
        folds.append(Fold(slice(0, val_start - gap), slice(val_start, test_start - gap), slice(test_start, test_start + test_len)))
    return folds


class Standardizer:
    """Feature scaler that is fit on training windows only."""

    def fit(self, x: np.ndarray) -> "Standardizer":
        flat = x.reshape(-1, x.shape[-1])
        self.mean_ = flat.mean(0)
        self.std_ = flat.std(0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)
