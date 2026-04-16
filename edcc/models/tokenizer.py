"""Stage 1: Cross-scale Tokenization.

Converts raw EEG windows (19ch x 400 samples) into token representations
using multi-scale temporal convolutions, event type embeddings, and
sinusoidal temporal positional encoding.
"""

import math
import torch
import torch.nn as nn


class MultiScaleTemporalConv(nn.Module):
    """Multi-scale 1D convolutions to capture different frequency bands.

    Branch 1 (k=15, 75ms): Beta/Alpha oscillations
    Branch 2 (k=40, 200ms): Alpha/Theta oscillations
    Branch 3 (k=100, 500ms): Delta/Theta oscillations

    Each branch: Conv1d → BatchNorm → GELU → AdaptiveAvgPool1d(1)
    Outputs are concatenated and projected to output_dim.
    """

    def __init__(self, output_dim: int = 128):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=40, padding=19),
            nn.BatchNorm1d(32),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100, padding=49),
            nn.BatchNorm1d(16),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
        )

        # 64 + 32 + 16 = 112 → output_dim
        self.projection = nn.Linear(112, output_dim)

    def forward(self, x):
        """
        Args:
            x: (N, 1, T) — single-channel signal segments.

        Returns:
            (N, output_dim) — token representation per channel per window.
        """
        b1 = self.branch1(x).squeeze(-1)    # (N, 64)
        b2 = self.branch2(x).squeeze(-1)    # (N, 32)
        b3 = self.branch3(x).squeeze(-1)    # (N, 16)
        cat = torch.cat([b1, b2, b3], dim=1)  # (N, 112)
        return self.projection(cat)            # (N, output_dim)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, positions):
        """
        Args:
            positions: (B, W) integer or float positions.
                If float (normalized 0-1), they are scaled to max_len.

        Returns:
            (B, W, d_model)
        """
        # Scale normalized positions to pe range
        max_len = self.pe.shape[0]
        indices = (positions * (max_len - 1)).long().clamp(0, max_len - 1)
        return self.pe[indices]


class CrossScaleTokenizer(nn.Module):
    """Stage 1: Cross-scale Tokenization module.

    Input: (B, W, 19, 400) — batch of windowed EEG
    Output: (B, W, 19, d_model) — per-channel per-window tokens

    Components:
        - Multi-scale temporal convolution (per channel)
        - Event type embedding
        - Sinusoidal temporal positional encoding
    """

    def __init__(
        self,
        d_model: int = 128,
        num_event_types: int = 6,
        window_size: int = 400,
    ):
        super().__init__()
        self.d_model = d_model

        self.temporal_conv = MultiScaleTemporalConv(output_dim=d_model)
        self.event_embed = nn.Embedding(num_event_types, d_model)
        self.temporal_pe = SinusoidalPE(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, windows, event_types, positions):
        """
        Args:
            windows: (B, W, 19, T) — raw EEG windows.
            event_types: (B, W) — event type per window.
            positions: (B, W) — normalized temporal positions.

        Returns:
            tokens: (B, W, 19, d_model) — per-channel per-window tokens.
        """
        B, W, C, T = windows.shape

        # Reshape for efficient batch processing: (B*W*C, 1, T)
        x = windows.reshape(B * W * C, 1, T)

        # Multi-scale temporal convolution
        tokens = self.temporal_conv(x)        # (B*W*C, d_model)
        tokens = tokens.reshape(B, W, C, self.d_model)  # (B, W, C, d_model)

        # Add event type embedding: (B, W, d_model) → broadcast to all channels
        event_emb = self.event_embed(event_types)  # (B, W, d_model)
        tokens = tokens + event_emb.unsqueeze(2)   # (B, W, C, d_model)

        # Add temporal positional encoding
        pos_emb = self.temporal_pe(positions)       # (B, W, d_model)
        tokens = tokens + pos_emb.unsqueeze(2)      # (B, W, C, d_model)

        tokens = self.layer_norm(tokens)

        return tokens
