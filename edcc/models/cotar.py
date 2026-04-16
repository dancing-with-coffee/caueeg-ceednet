"""Stage 2 Part 1: Core Token Aggregation-Redistribution (CoTAR).

Aggregates 19 channel tokens into a single core token per window,
then redistributes the processed core token back to channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoTAR(nn.Module):
    """Core Token Aggregation and Redistribution module.

    Aggregation: channel tokens → core token via attention-weighted sum.
    Redistribution: processed core token → channel updates via stored attention weights.

    Args:
        d_model: Channel token dimension.
        core_dim: Core token dimension (bottleneck).
        num_channels: Number of EEG channels (19).
    """

    def __init__(self, d_model: int = 128, core_dim: int = 64, num_channels: int = 19):
        super().__init__()
        self.d_model = d_model
        self.core_dim = core_dim

        # Aggregation projections
        self.q_proj = nn.Linear(d_model, core_dim)    # query from mean token
        self.k_proj = nn.Linear(d_model, core_dim)    # key from channel tokens
        self.v_proj = nn.Linear(d_model, core_dim)    # value from channel tokens

        self.scale = core_dim ** -0.5

        # Redistribution projection
        self.redist_proj = nn.Linear(core_dim, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model + core_dim, d_model),
            nn.Sigmoid(),
        )

    def aggregate(self, tokens):
        """Aggregate channel tokens into core tokens.

        Args:
            tokens: (B, W, C, d_model) — per-channel per-window tokens.

        Returns:
            core_tokens: (B, W, core_dim) — one core token per window.
            attn_weights: (B, W, C) — attention weights for redistribution.
        """
        B, W, C, D = tokens.shape

        # Initial core token: mean across channels
        core_init = tokens.mean(dim=2)           # (B, W, d_model)

        # Query from core, Key/Value from channels
        q = self.q_proj(core_init)                # (B, W, core_dim)
        k = self.k_proj(tokens)                   # (B, W, C, core_dim)
        v = self.v_proj(tokens)                   # (B, W, C, core_dim)

        # Attention: q (B,W,1,Dc) @ k^T (B,W,Dc,C) → (B,W,1,C)
        attn_scores = torch.einsum("bwd,bwcd->bwc", q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, W, C)

        # Weighted sum of values
        core_tokens = torch.einsum("bwc,bwcd->bwd", attn_weights, v)  # (B, W, core_dim)

        return core_tokens, attn_weights

    def redistribute(self, tokens, core_processed, attn_weights):
        """Redistribute processed core tokens back to channel tokens.

        Args:
            tokens: (B, W, C, d_model) — original channel tokens.
            core_processed: (B, W, core_dim) — core tokens after Mamba processing.
            attn_weights: (B, W, C) — attention weights from aggregation.

        Returns:
            updated_tokens: (B, W, C, d_model) — channel tokens updated with core info.
        """
        B, W, C, D = tokens.shape

        # Project core back to d_model
        core_proj = self.redist_proj(core_processed)  # (B, W, d_model)

        # Weight core by attention and expand to channels
        weighted_core = attn_weights.unsqueeze(-1) * core_proj.unsqueeze(2)  # (B, W, C, d_model)

        # Gated residual update
        gate_input = torch.cat([
            tokens,
            core_processed.unsqueeze(2).expand(-1, -1, C, -1),
        ], dim=-1)  # (B, W, C, d_model + core_dim)

        g = self.gate(gate_input)  # (B, W, C, d_model)

        updated = tokens + g * weighted_core

        return updated
