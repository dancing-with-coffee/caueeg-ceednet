"""Stage 2 Part 2: Event-Conditioned Sequence Modeling for core token trajectory.

Three backends (auto-selected in priority order):
  1. Native Mamba (mamba-ssm) — fastest, requires compatible CUDA
  2. Pure-torch SSD (Structured State-space Duality) — no CUDA extensions needed
  3. GatedConvRNN — simplest fallback

All backends support event conditioning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def drop_path(x, drop_prob=0.0, training=False):
    """DropPath (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor

# Try to import native Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except (ImportError, Exception):
    MAMBA_AVAILABLE = False


# ─── Pure-torch SSD implementation (from Mamba-2 paper, Listing 1) ───

def segsum(x):
    """Stable segment sum for causal masking."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_scan(X, A, B, C, block_len=16):
    """Pure-torch Structured State-space Duality scan.

    Args:
        X: (B, L, H, D) — input
        A: (B, L, H) — log decay rates
        B: (B, L, H, N) — input-to-state
        C: (B, L, H, N) — state-to-output
        block_len: chunk size for block-decomposition

    Returns:
        Y: (B, L, H, D) — output
    """
    B_sz, L, H, D = X.shape
    # Pad sequence length (dim=1) to multiple of block_len
    pad_len = (block_len - L % block_len) % block_len
    if pad_len > 0:
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len))     # (B, L+pad, H, D)
        A = F.pad(A, (0, 0, 0, pad_len))             # (B, L+pad, H)
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))      # (B, L+pad, H, N)
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))      # (B, L+pad, H, N)

    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len)
                   for x in (X, A, B, C)]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # Intra-chunk (diagonal blocks)
    L_mat = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L_mat, X)

    # State at chunk boundaries
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # Inter-chunk recurrence
    initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]

    # State → output
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # Remove padding
    if pad_len > 0:
        Y = Y[:, :L]

    return Y


class SSDBlock(nn.Module):
    """Pure-torch SSD block with event conditioning.

    Uses Structured State-space Duality (Mamba-2 algorithm) implemented
    entirely in PyTorch — no custom CUDA kernels needed.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 n_heads=4, event_embed_dim=16, drop_path_rate=0.0):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state
        self.n_heads = n_heads
        self.drop_path_rate = drop_path_rate
        assert d_inner % n_heads == 0
        self.head_dim = d_inner // n_heads

        # Input projection with event conditioning
        self.in_proj = nn.Linear(d_model + event_embed_dim, d_inner * 2, bias=False)

        # Local causal convolution
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(n_heads))  # log decay
        self.B_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.dt_proj = nn.Linear(d_inner, n_heads, bias=True)

        # Output
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, event_emb):
        """
        Args:
            x: (B, L, d_model)
            event_emb: (B, L, event_embed_dim)
        Returns:
            (B, L, d_model)
        """
        residual = x
        B_sz, L, D = x.shape

        # Project with event conditioning
        augmented = torch.cat([x, event_emb], dim=-1)
        xz = self.in_proj(augmented)  # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal local convolution
        x_conv = x_path.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM parameters
        dt = F.softplus(self.dt_proj(x_conv))  # (B, L, n_heads)
        A = -torch.exp(self.A_log.float())  # (n_heads,)
        A = dt * A  # (B, L, n_heads) — discretized decay

        B_ssm = self.B_proj(x_conv).reshape(B_sz, L, self.n_heads, self.d_state)
        C_ssm = self.C_proj(x_conv).reshape(B_sz, L, self.n_heads, self.d_state)
        X_ssm = x_conv.reshape(B_sz, L, self.n_heads, self.head_dim)

        # Run SSD scan
        Y = ssd_scan(X_ssm, A, B_ssm, C_ssm, block_len=min(16, L))
        Y = Y.reshape(B_sz, L, self.d_inner)

        # Gated output
        Y = Y * F.silu(z)
        out = self.out_proj(Y)

        return self.norm(residual + drop_path(out, self.drop_path_rate, self.training))


# ─── GatedConvRNN fallback ───

class GatedConvRNNBlock(nn.Module):
    """Mamba-inspired block with local conv + gating + GRU."""

    def __init__(self, d_model, d_conv=4, expand=2, event_embed_dim=16, drop_path_rate=0.0, **kwargs):
        super().__init__()
        d_inner = d_model * expand
        self.drop_path_rate = drop_path_rate

        self.in_proj = nn.Linear(d_model + event_embed_dim, d_inner * 2, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )
        self.rnn = nn.GRU(d_inner, d_inner, batch_first=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, event_emb):
        residual = x
        augmented = torch.cat([x, event_emb], dim=-1)
        xz = self.in_proj(augmented)
        x_path, z = xz.chunk(2, dim=-1)

        x_conv = x_path.transpose(1, 2)
        x_conv = self.conv(x_conv)[:, :, :x_path.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        x_rnn, _ = self.rnn(x_conv)
        y = x_rnn * F.silu(z)
        out = self.out_proj(y)

        return self.norm(residual + drop_path(out, self.drop_path_rate, self.training))


# ─── Main module ───

class EventConditionedMamba(nn.Module):
    """Multi-layer Event-Conditioned sequence model for core token trajectory.

    Auto-selects backend: Mamba > SSD (pure-torch) > GatedConvRNN.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        event_embed_dim: int = 16,
        num_event_types: int = 6,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.event_embed = nn.Embedding(num_event_types, event_embed_dim)

        # Stochastic depth: linearly increasing drop rate per layer
        dpr = [drop_path_rate * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Auto-select backend
        if MAMBA_AVAILABLE:
            self.backend = "mamba"
            self.layers = nn.ModuleList([
                _MambaBlockWrapper(d_model, d_state, d_conv, expand, event_embed_dim)
                for _ in range(num_layers)
            ])
        else:
            self.backend = "ssd"
            n_heads = max(1, d_model // 16)
            d_inner = d_model * expand
            while d_inner % n_heads != 0:
                n_heads -= 1
            self.layers = nn.ModuleList([
                SSDBlock(d_model, d_state, d_conv, expand, n_heads, event_embed_dim,
                         drop_path_rate=dpr[i])
                for i in range(num_layers)
            ])
            print(f"  [EDCC] Using SSD backend (pure-torch, {n_heads} heads, "
                  f"drop_path={drop_path_rate:.2f})")

        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, core_tokens, event_types, padding_mask=None):
        event_emb = self.event_embed(event_types)

        x = core_tokens
        for layer in self.layers:
            x = layer(x, event_emb)
            x = self.dropout(x)

        x = self.final_norm(x)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()

        return x


class _MambaBlockWrapper(nn.Module):
    """Wrapper for native Mamba with event conditioning."""

    def __init__(self, d_model, d_state, d_conv, expand, event_embed_dim):
        super().__init__()
        self.input_proj = nn.Linear(d_model + event_embed_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x, event_emb):
        augmented = torch.cat([x, event_emb], dim=-1)
        augmented = self.input_proj(augmented)
        residual = augmented
        return residual + self.mamba(self.norm(augmented))
