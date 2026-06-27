#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_embedding.py

Official D3 embedding layer.

D3 representation:
  - local numeric token signal: offset interpolation between bin embeddings
  - continuous signal: raw_scaled value in [0,1]
  - fusion: feature-wise FiLM/multiply

Input to forward:
  tokens: LongTensor [B, F]
  values: FloatTensor [B, F, 3]
    values[..., 0] = offset within bin
    values[..., 1] = raw_scaled continuous value
    values[..., 2] = continuous mask, usually 1 for D3

Output:
  cell_embeddings: FloatTensor [B, F, value_dim + feature_dim]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class D3OffsetFiLMEmbedding(nn.Module):
    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        value_dim: int,
        feature_dim: int,
        gate_init: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_bins <= 1:
            raise ValueError("num_bins must be > 1")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if value_dim <= 0 or feature_dim <= 0:
            raise ValueError("value_dim and feature_dim must be positive")

        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.init_std = float(init_std)

        # num_bins + 1 is required because interpolation at last bin uses b+1.
        self.bin_embedding = nn.Embedding(self.num_bins + 1, self.value_dim)
        self.feature_embedding = nn.Embedding(self.n_features, self.feature_dim)

        self.gamma_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.beta_proj = nn.Sequential(
            nn.Linear(1, self.value_dim),
            nn.GELU(),
            nn.Linear(self.value_dim, self.value_dim),
        )
        self.cont_gate_logit = nn.Parameter(torch.full((self.n_features, 1), float(gate_init)))

        nn.init.normal_(self.bin_embedding.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.feature_embedding.weight, mean=0.0, std=self.init_std)
        for net in [self.gamma_proj, self.beta_proj]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        self.register_buffer("default_feature_ids", torch.arange(self.n_features, dtype=torch.long), persistent=False)

    def local_interp(self, bin_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        b0 = bin_ids.long().clamp(0, self.num_bins - 1)
        b1 = b0 + 1
        e0 = self.bin_embedding(b0)
        e1 = self.bin_embedding(b1)
        return (1.0 - offset) * e0 + offset * e1

    def forward(self, tokens: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [B,F], got {tuple(tokens.shape)}")
        if values.ndim != 3 or values.shape[-1] < 3:
            raise ValueError(f"values must be [B,F,3], got {tuple(values.shape)}")
        if tuple(values.shape[:2]) != tuple(tokens.shape):
            raise ValueError(f"values shape must match tokens, got {tuple(values.shape)} vs {tuple(tokens.shape)}")

        vals = values.to(dtype=torch.float32, device=tokens.device)
        offset = vals[..., 0:1].clamp(0.0, 1.0)
        cont = vals[..., 1:2].clamp(0.0, 1.0)
        mask = vals[..., 2:3].clamp(0.0, 1.0)

        B, F = tokens.shape
        local = self.local_interp(tokens, offset)
        gamma = torch.tanh(self.gamma_proj(cont))
        beta = self.beta_proj(cont)

        gate = torch.sigmoid(self.cont_gate_logit).to(device=tokens.device).unsqueeze(0).expand(B, F, 1)
        g = mask * gate
        value_emb = local * (1.0 + g * gamma) + g * beta

        feature_ids = self.default_feature_ids.to(device=tokens.device).unsqueeze(0).expand(B, F)
        feature_emb = self.feature_embedding(feature_ids)
        return torch.cat([value_emb, feature_emb], dim=-1)

    def gate_summary(self) -> Dict[str, float]:
        with torch.no_grad():
            g = torch.sigmoid(self.cont_gate_logit.detach().cpu()).numpy()
        return {
            "cont_gate_min": float(g.min()),
            "cont_gate_max": float(g.max()),
            "cont_gate_mean": float(g.mean()),
            "cont_gate_std": float(g.std()),
        }
