"""Forecast metrics and the walk-forward comparison."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .baselines import RidgeAR, ZeroForecast
from .data import Standardizer, walk_forward_folds
from .train import predict, train_transformer


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    ic = spearmanr(y, p).statistic if np.std(p) > 0 else 0.0
    nz = y != 0
    return {
        "mse": float(np.mean((y - p) ** 2)),
        "ic": float(ic),
        "hit_rate": float(np.mean(np.sign(p[nz]) == np.sign(y[nz]))) if np.std(p) > 0 else 0.5,
    }


def walk_forward(x: np.ndarray, y: np.ndarray, n_folds: int = 4, gap: int = 1, epochs: int = 30) -> dict[str, dict]:
    """Fit every model on each expanding train window; score on the following unseen block."""
    preds = {"zero": [], "ridge": [], "transformer": []}
    truth = []
    for fold in walk_forward_folds(len(x), n_folds=n_folds, gap=gap):
        sc = Standardizer().fit(x[fold.train])
        xtr, xva, xte = (sc.transform(x[s]) for s in (fold.train, fold.val, fold.test))
        ytr, yva, yte = y[fold.train], y[fold.val], y[fold.test]
        preds["zero"].append(ZeroForecast().fit(xtr, ytr).predict(xte))
        preds["ridge"].append(RidgeAR().fit(xtr, ytr).predict(xte))
        model = train_transformer(xtr, ytr, xva, yva, epochs=epochs)
        preds["transformer"].append(predict(model, xte))
        truth.append(yte)
    y_all = np.concatenate(truth)
    return {k: metrics(y_all, np.concatenate(v)) for k, v in preds.items()}
