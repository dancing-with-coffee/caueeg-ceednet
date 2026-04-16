"""Age-Conditioned Classification Head.

Uses age-conditioned layer normalization and an optional age adversarial
branch with gradient reversal for debiasing.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversal(Function):
    """Gradient reversal layer for adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_rev):
        ctx.lambda_rev = lambda_rev
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_rev * grad_output, None


def gradient_reversal(x, lambda_rev=1.0):
    return _GradientReversal.apply(x, lambda_rev)


class AgeConditionedClassifier(nn.Module):
    """Classification head with age-conditioned normalization.

    Components:
        - Age binning (10-year intervals) + embedding → conditional LayerNorm
        - MLP classifier: d_model → hidden → num_classes
        - Optional: age adversarial branch with gradient reversal

    Args:
        d_model: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        num_classes: Number of output classes.
        num_age_bins: Number of age bins for conditioning.
        age_min: Minimum age for binning.
        age_bin_width: Width of each age bin in years.
        dropout: Dropout rate.
        age_adversarial: Whether to use age adversarial branch.
        lambda_rev: Gradient reversal strength.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 3,
        num_age_bins: int = 10,
        age_min: float = 40.0,
        age_bin_width: float = 7.0,
        dropout: float = 0.3,
        age_adversarial: bool = True,
        lambda_rev: float = 0.1,
        use_coral: bool = False,
    ):
        super().__init__()
        self.num_age_bins = num_age_bins
        self.age_min = age_min
        self.age_bin_width = age_bin_width
        self.lambda_rev = lambda_rev
        self.use_coral = use_coral
        # CORAL outputs K-1 cumulative logits instead of K class logits
        output_dim = num_classes - 1 if use_coral else num_classes

        # Age-conditioned layer norm
        # Embedding → (gamma, beta) for affine transform
        self.age_embed = nn.Embedding(num_age_bins, d_model * 2)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # Classification MLP — age is concatenated (CeedNet style) + conditioned via LN
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),  # +1 for age concat
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Age adversarial branch
        self.age_adversarial = age_adversarial
        if age_adversarial:
            self.age_predictor = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _age_to_bin(self, age):
        """Convert age values to bin indices."""
        bins = ((age - self.age_min) / self.age_bin_width).long()
        return bins.clamp(0, self.num_age_bins - 1)

    def forward(self, z, age):
        """
        Args:
            z: (B, d_model) — graph-level representation from GCN.
            age: (B,) — patient ages.

        Returns:
            logits: (B, num_classes)
            age_pred: (B, 1) or None — predicted age (for adversarial loss)
        """
        # Age-conditioned layer norm
        age_bin = self._age_to_bin(age)
        gamma_beta = self.age_embed(age_bin)   # (B, 2*d_model)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        z = self.layer_norm(z) * (1 + gamma) + beta

        # Age adversarial prediction (before age concat to avoid info leak)
        age_pred = None
        if self.age_adversarial and self.training:
            z_rev = gradient_reversal(z, self.lambda_rev)
            age_pred = self.age_predictor(z_rev)  # (B, 1)

        # Concat normalized age (CeedNet style — simple but effective)
        age_norm = ((age - 70.0) / 15.0).unsqueeze(-1)  # (B, 1)
        z = torch.cat([z, age_norm], dim=-1)  # (B, d_model + 1)

        # Classification
        logits = self.classifier(z)

        return logits, age_pred
