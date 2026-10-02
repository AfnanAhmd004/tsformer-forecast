"""Training with early stopping on a validation set."""
from __future__ import annotations

import copy

import numpy as np
import torch

from .model import TSTransformer


def train_transformer(x_tr, y_tr, x_va, y_va, epochs: int = 30, lr: float = 1e-3, batch: int = 128,
                      patience: int = 5, seed: int = 0, **model_kw) -> TSTransformer:
    torch.manual_seed(seed)
    model = TSTransformer(n_features=x_tr.shape[-1], lookback=x_tr.shape[1], **model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv, yv = torch.from_numpy(x_va), torch.from_numpy(y_va)
    y_scale = float(yt.std()) + 1e-8  # train on standardised targets for stable optimisation

    best, best_loss, bad = copy.deepcopy(model.state_dict()), np.inf, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx] / y_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(xv), yv / y_scale).item()
        if val < best_loss - 1e-6:
            best_loss, best, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best)
    model.y_scale = y_scale
    return model


def predict(model: TSTransformer, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (model(torch.from_numpy(x)) * model.y_scale).numpy()
