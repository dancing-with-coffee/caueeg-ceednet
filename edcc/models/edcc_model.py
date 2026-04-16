"""Full EDCC Model: Event-Aware Dynamic Centralized Cross-scale EEG Model.

Assembles all stages:
    Stage 1: Cross-scale Tokenization
    Stage 2: CoTAR + Event-Conditioned Mamba
    Stage 3: Region-Level GCN
    Classification Head: Age-Conditioned MLP
"""

import torch
import torch.nn as nn

from .tokenizer import CrossScaleTokenizer
from .cotar import CoTAR
from .mamba_core import EventConditionedMamba
from .gcn import RegionGCN
from .classifier import AgeConditionedClassifier


class EDCCModel(nn.Module):
    """Full EDCC model for EEG-based dementia classification.

    Args:
        d_model: Token dimension (Stage 1 output / Stage 2-3 working dim).
        core_dim: Core token dimension (CoTAR bottleneck).
        num_channels: Number of EEG channels.
        num_classes: Number of output classes.
        window_size: Window size in samples.
        num_event_types: Number of event types.
        mamba_layers: Number of Mamba layers.
        mamba_d_state: SSM state dimension.
        gcn_layers: Number of GCN layers.
        gcn_hidden: GCN hidden dimension.
        dropout: Dropout rate.
        age_adversarial: Enable age adversarial training.
        lambda_rev: Gradient reversal strength.
    """

    def __init__(
        self,
        d_model: int = 128,
        core_dim: int = 64,
        num_channels: int = 19,
        num_classes: int = 3,
        window_size: int = 400,
        num_event_types: int = 6,
        mamba_layers: int = 4,
        mamba_d_state: int = 16,
        gcn_layers: int = 2,
        gcn_hidden: int = 128,
        dropout: float = 0.1,
        age_adversarial: bool = True,
        lambda_rev: float = 0.1,
        use_coral: bool = False,
    ):
        super().__init__()
        self.use_coral = use_coral

        # Stage 1: Cross-scale Tokenization
        self.tokenizer = CrossScaleTokenizer(
            d_model=d_model,
            num_event_types=num_event_types,
            window_size=window_size,
        )

        # Stage 2: CoTAR + Mamba
        self.cotar = CoTAR(
            d_model=d_model,
            core_dim=core_dim,
            num_channels=num_channels,
        )

        self.mamba = EventConditionedMamba(
            d_model=core_dim,
            d_state=mamba_d_state,
            num_layers=mamba_layers,
            num_event_types=num_event_types,
            dropout=dropout,
        )

        # Stage 3: Region-Level GCN
        self.gcn = RegionGCN(
            d_model=d_model,
            hidden_dim=gcn_hidden,
            num_gcn_layers=gcn_layers,
            dropout=dropout,
        )

        # Classification Head
        self.classifier = AgeConditionedClassifier(
            d_model=d_model,
            hidden_dim=d_model // 2,
            num_classes=num_classes,
            age_adversarial=age_adversarial,
            lambda_rev=lambda_rev,
            dropout=dropout,
            use_coral=use_coral,
        )

        # Window-level auxiliary classifier (standard CE, not CORAL)
        self.window_classifier = nn.Sequential(
            nn.Linear(core_dim, core_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(core_dim, num_classes),
        )

        # DropPath for regularization
        self.drop_path_rate = dropout

    def forward(self, windows, event_types, positions, age, padding_mask=None):
        """
        Args:
            windows: (B, W, 19, T) — raw EEG windows.
            event_types: (B, W) — event type per window.
            positions: (B, W) — normalized temporal positions.
            age: (B,) — patient ages.
            padding_mask: (B, W) — True for valid positions.

        Returns:
            logits: (B, num_classes)
            aux: dict with age_pred and window_logits
        """
        # Stage 1: Tokenize
        tokens = self.tokenizer(windows, event_types, positions)
        # tokens: (B, W, 19, d_model)

        # Stage 2: CoTAR aggregate → Mamba → CoTAR redistribute
        core_tokens, attn_weights = self.cotar.aggregate(tokens)
        # core_tokens: (B, W, core_dim)

        core_processed = self.mamba(core_tokens, event_types, padding_mask)
        # core_processed: (B, W, core_dim)

        h = self.cotar.redistribute(tokens, core_processed, attn_weights)
        # h: (B, W, 19, d_model)

        # Stage 3: Region pooling + GCN
        graph_repr = self.gcn(h, padding_mask)
        # graph_repr: (B, d_model)

        # Classification
        logits, age_pred = self.classifier(graph_repr, age)

        # Window-level auxiliary: classify each window independently
        window_logits = self.window_classifier(core_processed)
        # window_logits: (B, W, num_classes)

        aux = {"age_pred": age_pred, "window_logits": window_logits}
        return logits, aux

    def count_parameters(self):
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
