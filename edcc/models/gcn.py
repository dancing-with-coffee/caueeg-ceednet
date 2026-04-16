"""Stage 3: Region-Level Graph Convolutional Network.

Pools 19-channel tokens into 5 brain regions, then applies GCN
for spatial reasoning across regions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel-to-region mapping (indices into 19-channel array)
REGION_CHANNELS = {
    "Frontal":   [0, 5, 1, 6, 10, 13, 16],   # Fp1,Fp2,F3,F4,F7,F8,FZ
    "Central":   [2, 7, 17],                    # C3,C4,CZ
    "Parietal":  [3, 8, 18],                    # P3,P4,PZ
    "Temporal":  [11, 14, 12, 15],              # T3,T4,T5,T6
    "Occipital": [4, 9],                        # O1,O2
}
REGION_NAMES = list(REGION_CHANNELS.keys())
NUM_REGIONS = len(REGION_NAMES)


def _build_adjacency():
    """Build a fixed adjacency matrix based on anatomical proximity.

    Connections:
        Frontal-Central, Frontal-Temporal,
        Central-Parietal, Central-Temporal,
        Parietal-Temporal, Parietal-Occipital
    All nodes have self-loops.
    """
    adj = torch.eye(NUM_REGIONS)

    edges = [
        (0, 1),  # Frontal-Central
        (0, 3),  # Frontal-Temporal
        (1, 2),  # Central-Parietal
        (1, 3),  # Central-Temporal
        (2, 3),  # Parietal-Temporal
        (2, 4),  # Parietal-Occipital
    ]
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = adj.sum(dim=1)
    d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degree))
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt

    return adj_norm


def _compute_laplacian_pe(adj, pe_dim=4):
    """Compute Laplacian positional encoding for the graph."""
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # Take the smallest non-trivial eigenvectors (skip the first constant one)
    pe = eigenvectors[:, 1 : pe_dim + 1]  # (num_nodes, pe_dim)

    # Pad if graph is too small
    if pe.shape[1] < pe_dim:
        pe = F.pad(pe, (0, pe_dim - pe.shape[1]))

    return pe


class GCNLayer(nn.Module):
    """Simple GCN layer for a fixed small graph."""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        Args:
            x: (B, N, in_dim)
            adj: (N, N) normalized adjacency.
        Returns:
            (B, N, out_dim)
        """
        # Message passing: A @ X
        x = torch.matmul(adj, x)  # (B, N, in_dim)
        x = self.linear(x)        # (B, N, out_dim)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class RegionGCN(nn.Module):
    """Region-level GCN module.

    1. Pools 19 channel tokens → 5 region tokens (attention-weighted mean).
    2. Adds Laplacian positional encoding.
    3. Applies GCN layers.
    4. Mean-pools region nodes → graph-level representation.

    Args:
        d_model: Input channel token dimension.
        hidden_dim: GCN hidden dimension.
        num_gcn_layers: Number of GCN layers.
        lap_pe_dim: Laplacian PE dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 128,
        num_gcn_layers: int = 2,
        lap_pe_dim: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model

        # Precompute region channel indices as flat list for gather
        self._region_channel_list = [REGION_CHANNELS[r] for r in REGION_NAMES]

        # Learnable attention pooling per region
        self.region_attn = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(NUM_REGIONS)
        ])

        # Laplacian PE
        adj_raw = _build_adjacency()
        self.register_buffer("adj", _build_adjacency())
        lap_pe = _compute_laplacian_pe(adj_raw, lap_pe_dim)
        self.register_buffer("lap_pe", lap_pe)

        self.pe_proj = nn.Linear(lap_pe_dim, d_model)

        # GCN layers
        layers = []
        in_dim = d_model
        for i in range(num_gcn_layers):
            out_dim = hidden_dim if i < num_gcn_layers - 1 else d_model
            layers.append(GCNLayer(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.gcn_layers = nn.ModuleList(layers)

    def forward(self, channel_tokens, padding_mask=None):
        """
        Args:
            channel_tokens: (B, W, 19, d_model) — per-channel tokens after CoTAR redistribution.
            padding_mask: (B, W) — True for valid windows.

        Returns:
            graph_repr: (B, d_model) — graph-level representation.
        """
        B, W, C, D = channel_tokens.shape

        # 1. Time pooling: average across valid windows → (B, 19, d_model)
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, W, 1, 1)
            channel_tokens = channel_tokens * mask
            time_pooled = channel_tokens.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            time_pooled = channel_tokens.mean(dim=1)  # (B, 19, d_model)

        # 2. Region pooling: 19 channels → 5 regions via attention-weighted mean
        region_tokens = []
        for r_idx, ch_indices in enumerate(self._region_channel_list):
            region_ch = time_pooled[:, ch_indices, :]  # (B, n_ch, d_model)
            attn = self.region_attn[r_idx](region_ch)  # (B, n_ch, 1)
            attn = F.softmax(attn, dim=1)
            pooled = (attn * region_ch).sum(dim=1)     # (B, d_model)
            region_tokens.append(pooled)

        region_tokens = torch.stack(region_tokens, dim=1)  # (B, 5, d_model)

        # 3. Add Laplacian PE
        region_tokens = region_tokens + self.pe_proj(self.lap_pe)

        # 4. GCN layers
        x = region_tokens
        for layer in self.gcn_layers:
            x = layer(x, self.adj)

        # 5. Graph-level readout: mean pooling
        graph_repr = x.mean(dim=1)  # (B, d_model)

        return graph_repr
