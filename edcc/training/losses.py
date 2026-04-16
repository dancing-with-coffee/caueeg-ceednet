"""Loss functions for EDCC training.

Includes sample-level CE, subject-level CE (LEAD-style),
and optional age adversarial loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses learning on hard examples (low p_t), which helps with
    underrepresented classes like MCI and Dementia.
    """

    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights  # stored as list, converted per-forward

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) raw logits
            targets: (B,) class indices
        """
        # Apply label smoothing via soft targets
        num_classes = logits.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            smooth = F.one_hot(targets, num_classes).float()

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Class weights
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            w = w[targets].unsqueeze(1)  # (B, 1)
            focal_weight = focal_weight * w

        loss = -focal_weight * smooth * log_probs
        return loss.sum(dim=1).mean()


class RACELoss(nn.Module):
    """Rank-Aware Cross-Entropy (RACE) — novel loss combining CE sharpness
    with ordinal distance structure.

    Creates ordinal-aware soft targets where probability mass spreads
    according to class distance:

        q(j|k) = (1-ε_k) · δ(j,k) + ε_k · exp(-|j-k|/σ) / Z_k

    Supports two modes:
        1. Uniform RACE: same ε for all classes
        2. Adaptive RACE (A-RACE): class-dependent ε based on ordinal position
           - Endpoint classes (Normal, Dementia): higher ε → ordinal awareness
           - Middle classes (MCI): lower ε → sharp CE-like boundaries

    A-RACE rationale: Middle classes need sharp boundaries to avoid being
    absorbed by neighbors. Endpoint classes benefit from ordinal smoothing
    to suppress far-away misclassifications.

    Args:
        num_classes: Number of ordinal classes.
        epsilon: Smoothing rate (scalar or list of per-class values).
            If scalar: same ε for all classes (uniform RACE).
            If list: per-class ε (A-RACE), e.g. [0.20, 0.05, 0.20].
        sigma: Ordinal temperature — lower = sharper distance decay.
        class_weights: Optional per-class weights for loss weighting.
    """

    def __init__(self, num_classes=3, epsilon=0.15, sigma=0.8, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

        # Support per-class epsilon (A-RACE)
        if isinstance(epsilon, (list, tuple)):
            assert len(epsilon) == num_classes
            epsilons = epsilon
        else:
            epsilons = [epsilon] * num_classes

        # Pre-compute ordinal soft targets for each class
        ranks = torch.arange(num_classes, dtype=torch.float)

        # Ordinal distance distribution: exp(-|j-k|/sigma) / Z
        ordinal_dist = torch.zeros(num_classes, num_classes)
        for k in range(num_classes):
            distances = torch.abs(ranks - k)
            ordinal_dist[k] = torch.exp(-distances / sigma)
            ordinal_dist[k] /= ordinal_dist[k].sum()

        # Final targets: (1-ε_k) · one_hot + ε_k · ordinal_dist
        one_hot = torch.eye(num_classes)
        final_targets = torch.zeros(num_classes, num_classes)
        for k in range(num_classes):
            final_targets[k] = (1 - epsilons[k]) * one_hot[k] + epsilons[k] * ordinal_dist[k]

        self.register_buffer("target_matrix", final_targets)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K) raw logits
            targets: (B,) class indices
        Returns:
            scalar loss
        """
        # Look up soft targets for each sample
        soft_targets = self.target_matrix.to(logits.device)[targets]  # (B, K)

        log_probs = F.log_softmax(logits, dim=1)

        # Weighted CE with soft ordinal targets
        loss = -(soft_targets * log_probs).sum(dim=1)

        # Optional class weighting
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            loss = loss * w[targets]

        return loss.mean()

    def get_target_matrix(self):
        """Return the pre-computed soft target matrix for inspection."""
        return self.target_matrix


class OrdinalPenalty(nn.Module):
    """Ordinal distance penalty — penalizes predictions far from the true rank.

    Computes the expected ordinal distance: E[|Y_pred - Y_true|]
    where Y_pred is the expected rank from softmax probabilities.

    This supplements CE loss:
    - CE provides sharp class boundaries (good for MCI)
    - Ordinal penalty discourages Normal↔Dementia confusion (2-step error)
    """

    def __init__(self, num_classes=3):
        super().__init__()
        # Pre-compute rank values: [0, 1, 2]
        self.register_buffer("ranks", torch.arange(num_classes, dtype=torch.float))

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K) — standard softmax logits (NOT CORAL)
            targets: (B,) — ordinal class indices
        Returns:
            scalar loss — mean expected absolute ordinal distance
        """
        probs = F.softmax(logits, dim=1)  # (B, K)
        ranks = self.ranks.to(logits.device)

        # Expected rank: E[Y_pred] = sum_k (k * p(k))
        expected_rank = (probs * ranks.unsqueeze(0)).sum(dim=1)  # (B,)

        # Absolute ordinal distance to true rank
        true_rank = targets.float()
        ordinal_distance = torch.abs(expected_rank - true_rank)

        return ordinal_distance.mean()


class CORALLoss(nn.Module):
    """CORAL (Consistent Rank Logits) for ordinal regression.

    Decomposes K-class ordinal problem into K-1 binary tasks:
        P(Y > 0), P(Y > 1), ..., P(Y > K-2)

    Each binary task shares feature weights but has its own bias.
    Normal(0) → MCI(1) → Dementia(2) ordering is enforced.

    Reference: Cao, Mirjalili, Raschka (2020) "Rank consistent ordinal regression
    for neural networks with application to age estimation"
    """

    def __init__(self, num_classes=3, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, K-1) — cumulative logits (NOT softmax logits).
                    logits[:, k] = log-odds of P(Y > k)
            targets: (B,) — ordinal class indices {0, 1, ..., K-1}
        """
        num_tasks = self.num_classes - 1  # K-1 binary tasks

        # Create binary targets: target[k] = 1 if Y > k, else 0
        # Normal(0): [0, 0]  — not > 0, not > 1
        # MCI(1):    [1, 0]  — > 0, not > 1
        # Dementia(2): [1, 1]  — > 0, > 1
        binary_targets = torch.zeros(targets.size(0), num_tasks,
                                      device=targets.device, dtype=logits.dtype)
        for k in range(num_tasks):
            binary_targets[:, k] = (targets > k).float()

        # Binary cross-entropy for each task
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='none')

        # Optional class weighting
        if self.class_weights is not None:
            w = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
            sample_weights = w[targets].unsqueeze(1)  # (B, 1)
            loss = loss * sample_weights

        return loss.mean()

    @staticmethod
    def logits_to_probs(logits):
        """Convert CORAL cumulative logits to class probabilities.

        Args:
            logits: (B, K-1) cumulative logits

        Returns:
            probs: (B, K) class probabilities
        """
        cum_probs = torch.sigmoid(logits)  # P(Y > k) for k = 0, ..., K-2
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y = k) = P(Y > k-1) - P(Y > k)  for 0 < k < K-1
        # P(Y = K-1) = P(Y > K-2)
        probs = torch.zeros(logits.size(0), logits.size(1) + 1, device=logits.device)
        probs[:, 0] = 1 - cum_probs[:, 0]
        for k in range(1, logits.size(1)):
            probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
        probs[:, -1] = cum_probs[:, -1]
        # Clamp for numerical stability
        probs = probs.clamp(min=1e-7)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    @staticmethod
    def logits_to_pred(logits):
        """Convert CORAL logits to predicted class."""
        return CORALLoss.logits_to_probs(logits).argmax(dim=1)


class EDCCLoss(nn.Module):
    """Combined loss for EDCC training.

    L = alpha * L_sample + beta * L_subject + gamma * L_age

    Supports CrossEntropy, Focal Loss, and CORAL ordinal regression.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 0.1,
        label_smoothing: float = 0.1,
        num_classes: int = 3,
        class_weights: list = None,
        focal_gamma: float = 0.0,
        use_coral: bool = False,
        ordinal_weight: float = 0.0,
        use_race: bool = False,
        race_epsilon: float = 0.15,
        race_sigma: float = 0.8,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.use_coral = use_coral
        self.ordinal_weight = ordinal_weight
        self.use_race = use_race

        # Ordinal penalty (CE + ordinal hybrid)
        if ordinal_weight > 0 and not use_coral:
            self.ordinal_penalty = OrdinalPenalty(num_classes=num_classes)

        weight = None
        if class_weights:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weight", weight)

        # Loss function selection (priority: CORAL > RACE > Focal > CE)
        self.use_focal = False
        if use_coral:
            self.coral_loss = CORALLoss(num_classes=num_classes, class_weights=class_weights)
        elif use_race:
            self.race_loss = RACELoss(
                num_classes=num_classes, epsilon=race_epsilon, sigma=race_sigma,
                class_weights=class_weights,
            )
        elif focal_gamma > 0:
            self.use_focal = True
            self.focal_loss = FocalLoss(
                gamma=focal_gamma,
                class_weights=class_weights,
                label_smoothing=label_smoothing,
            )
        else:
            self.sample_ce = nn.CrossEntropyLoss(
                weight=weight, label_smoothing=label_smoothing
            )

    def forward(self, logits, targets, age_pred=None, age_true=None, serials=None,
                window_logits=None, padding_mask=None, window_aux_weight=0.3):
        """
        Args:
            logits: (B, num_classes) or (B, num_classes-1) for CORAL
            targets: (B,)
            age_pred: (B, 1) or None
            age_true: (B,) or None
            serials: list of str or None
            window_logits: (B, W, num_classes) or None
            padding_mask: (B, W) or None
            window_aux_weight: weight for window-level auxiliary loss

        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        # Sample-level loss (CORAL, RACE, Focal, or CE)
        if self.use_coral:
            loss_sample = self.coral_loss(logits, targets)
        elif self.use_race:
            loss_sample = self.race_loss(logits, targets)
        elif self.use_focal:
            loss_sample = self.focal_loss(logits, targets)
        elif self.class_weight is not None:
            w = self.class_weight.to(logits.device)
            loss_sample = F.cross_entropy(
                logits, targets, weight=w, label_smoothing=self.label_smoothing
            )
        else:
            loss_sample = self.sample_ce(logits, targets)

        total = self.alpha * loss_sample
        loss_dict = {"loss_sample": loss_sample.item()}

        # Ordinal distance penalty (hybrid CE + ordinal)
        if self.ordinal_weight > 0 and not self.use_coral and hasattr(self, 'ordinal_penalty'):
            loss_ord = self.ordinal_penalty(logits, targets)
            total = total + self.ordinal_weight * loss_ord
            loss_dict["loss_ordinal"] = loss_ord.item()

        # Subject-level CE (average logits per subject, then CE)
        if serials is not None and self.beta > 0:
            loss_subject = self._subject_level_ce(logits, targets, serials)
            total = total + self.beta * loss_subject
            loss_dict["loss_subject"] = loss_subject.item()

        # Age adversarial loss (normalize age to 0-1 range for stable MSE)
        if age_pred is not None and age_true is not None and self.gamma > 0:
            age_normalized = (age_true - 65.0) / 15.0
            loss_age = F.mse_loss(age_pred.squeeze(-1), age_normalized)
            total = total + self.gamma * loss_age
            loss_dict["loss_age"] = loss_age.item()

        # Window-level auxiliary loss: each window predicts the recording's class
        # This multiplies effective training samples by ~128x
        if window_logits is not None and window_aux_weight > 0 and padding_mask is not None:
            B, W, C = window_logits.shape
            # Expand targets to all windows: (B,) → (B*W,)
            window_targets = targets.unsqueeze(1).expand(-1, W).reshape(-1)
            flat_logits = window_logits.reshape(-1, C)
            flat_mask = padding_mask.reshape(-1)

            # Only compute loss on valid (non-padded) windows
            valid_logits = flat_logits[flat_mask]
            valid_targets = window_targets[flat_mask]

            if len(valid_logits) > 0:
                loss_window = F.cross_entropy(valid_logits, valid_targets,
                                               label_smoothing=self.label_smoothing)
                total = total + window_aux_weight * loss_window
                loss_dict["loss_window"] = loss_window.item()

        loss_dict["loss_total"] = total.item()
        return total, loss_dict

    def _subject_level_ce(self, logits, targets, serials):
        """Compute subject-level CE by averaging logits per subject."""
        # Group by serial
        serial_to_indices = {}
        for i, s in enumerate(serials):
            serial_to_indices.setdefault(s, []).append(i)

        subject_logits = []
        subject_targets = []

        for serial, indices in serial_to_indices.items():
            idx = torch.tensor(indices, device=logits.device)
            avg_logit = logits[idx].mean(dim=0)
            target = targets[idx[0]]  # All same subject = same label
            subject_logits.append(avg_logit)
            subject_targets.append(target)

        if not subject_logits:
            return torch.tensor(0.0, device=logits.device)

        subject_logits = torch.stack(subject_logits)
        subject_targets = torch.stack(subject_targets)

        return F.cross_entropy(subject_logits, subject_targets)
