#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_model.py

Official D3 C2/D3 model.

D3 = offset interpolation + raw_scaled FiLM/multiply fusion, followed by a
feature-level TransformerEncoder and CLS classifier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

try:
    import config as CFG
except Exception:
    CFG = None

_embedding_path = Path(__file__).resolve().with_name("02_embedding.py")
_embedding_spec = importlib.util.spec_from_file_location("_src_baocao_02_embedding", _embedding_path)
_embedding_mod = importlib.util.module_from_spec(_embedding_spec)
assert _embedding_spec is not None and _embedding_spec.loader is not None
_embedding_spec.loader.exec_module(_embedding_mod)
D3OffsetFiLMEmbedding = _embedding_mod.D3OffsetFiLMEmbedding


def cfg(name: str, default):
    return getattr(CFG, name, default) if CFG is not None else default


class D3C2D3Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_bins: int,
        n_features: int,
        num_classes: int,
        value_dim: int | None = None,
        feature_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        dropout: float | None = None,
        classifier_hidden_dim: int | None = None,
        classifier_dropout: float | None = None,
        norm_first: bool | None = None,
        gate_init: float | None = None,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        if value_dim is None:
            value_dim = int(cfg("VALUE_EMBED_DIM", 32))
        if feature_dim is None:
            feature_dim = int(cfg("FEATURE_EMBED_DIM", 32))
        if hidden_dim is None:
            hidden_dim = int(cfg("MODEL_HIDDEN_DIM", 128))
        if num_layers is None:
            num_layers = int(cfg("MODEL_NUM_LAYERS", 3))
        if num_heads is None:
            num_heads = int(cfg("MODEL_NUM_HEADS", 4))
        if dropout is None:
            dropout = float(cfg("MODEL_DROPOUT", 0.1))
        if classifier_hidden_dim is None:
            classifier_hidden_dim = int(cfg("CLASSIFIER_HIDDEN_DIM", hidden_dim))
        if classifier_dropout is None:
            classifier_dropout = float(cfg("CLASSIFIER_DROPOUT", dropout))
        if norm_first is None:
            norm_first = bool(cfg("TRANSFORMER_NORM_FIRST", True))
        if gate_init is None:
            gate_init = float(cfg("GATE_INIT", 0.0))
        if activation is None:
            activation = str(cfg("MODEL_ACTIVATION", "gelu"))

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim must be divisible by num_heads: {hidden_dim}/{num_heads}")

        self.run_id = "D3"
        self.num_bins = int(num_bins)
        self.n_features = int(n_features)
        self.num_classes = int(num_classes)
        self.value_dim = int(value_dim)
        self.feature_dim = int(feature_dim)
        self.cell_dim = int(value_dim + feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.model_spec = {
            "run_id": "D3",
            "local": "offset_interpolation",
            "continuous_source": "raw_scaled",
            "fusion": "raw_film",
            "description": "offset interpolation + raw FiLM/multiply fusion",
        }

        self.embedding = D3OffsetFiLMEmbedding(
            num_bins=int(num_bins),
            n_features=int(n_features),
            value_dim=int(value_dim),
            feature_dim=int(feature_dim),
            gate_init=float(gate_init),
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, self.hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation=str(activation),
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, int(classifier_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(classifier_dropout)),
            nn.Linear(int(classifier_hidden_dim), int(num_classes)),
        )

    def forward(self, tokens: torch.Tensor, values: torch.Tensor, *, return_info: bool = False):
        cell_emb = self.embedding(tokens, values)
        x = self.input_proj(cell_emb)
        B = x.shape[0]
        cls = self.cls_token.expand(B, 1, self.hidden_dim)
        x = torch.cat([cls, x], dim=1)
        encoded = self.encoder(x)
        cls_out = encoded[:, 0, :]
        logits = self.classifier(cls_out)
        if return_info:
            return logits, {
                "cell_emb_shape": tuple(cell_emb.shape),
                "encoded_shape": tuple(encoded.shape),
                "cls_out_shape": tuple(cls_out.shape),
                "cell_dim": self.cell_dim,
                "num_bins": self.num_bins,
                "run_id": self.run_id,
                "spec": self.model_spec,
            }
        return logits

    def embedding_extra_summary(self) -> Dict[str, float]:
        return self.embedding.gate_summary()
