"""A compact Transformer encoder for univariate/multivariate sequence regression."""
from __future__ import annotations

import torch
from torch import nn


class TSTransformer(nn.Module):
    def __init__(self, n_features: int, lookback: int, d_model: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, n_features)
        h = self.encoder(self.input(x) + self.pos)
        return self.head(self.norm(h[:, -1])).squeeze(-1)  # read out from the last time step
