from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class LinearPolicy(nn.Module):
    """Linear scoring model for portfolio construction.
    
    Computes scores as: score_i = x_i' theta + b
    """
    
    def __init__(self, n_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=bias)
        
        # Initialize near zero
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.linear(X).squeeze(-1)


class MLPPolicy(nn.Module):
    """MLP scoring model for portfolio construction.
    
    Computes scores as: score_i = f(x_i) where f is a neural network.
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: Sequence[int] = (32, 16, 8),
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev_dim = n_features

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())

            if dropout > 0 and i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)
