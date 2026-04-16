"""EDCC-specific data augmentations applied at the batch level."""

import torch
import numpy as np


def apply_mixup(batch, alpha=0.2):
    """Apply mixup augmentation at the recording level.

    Mixes pairs of samples within the batch using Beta distribution.

    Args:
        batch: Dict from edcc_collate_fn.
        alpha: Beta distribution parameter. 0 = disabled.

    Returns:
        Modified batch with mixed windows, and (targets_a, targets_b, lam) for loss.
    """
    if alpha <= 0:
        return batch, None

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

    B = batch["windows"].size(0)
    index = torch.randperm(B, device=batch["windows"].device)

    batch["windows"] = lam * batch["windows"] + (1 - lam) * batch["windows"][index]
    batch["age"] = lam * batch["age"] + (1 - lam) * batch["age"][index]

    mixup_info = {
        "targets_a": batch["class_label"],
        "targets_b": batch["class_label"][index],
        "lam": lam,
    }

    return batch, mixup_info


def apply_augmentations(batch, noise_std=0.05, window_drop_rate=0.1, channel_drop_rate=0.1):
    """Apply augmentations to a collated batch (in-place).

    Args:
        batch: Dict from edcc_collate_fn.
        noise_std: Standard deviation of additive Gaussian noise.
        window_drop_rate: Fraction of non-transition windows to randomly drop.
        channel_drop_rate: Fraction of channels to randomly zero out.

    Returns:
        Modified batch dict.
    """
    windows = batch["windows"]          # (B, W, 19, T)
    padding_mask = batch["padding_mask"]  # (B, W)
    event_types = batch["event_types"]   # (B, W)

    # 1. Additive Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(windows) * noise_std
        # Only add noise to valid (non-padded) windows
        noise = noise * padding_mask.unsqueeze(-1).unsqueeze(-1).float()
        windows = windows + noise

    # 2. Window dropout — randomly mask non-transition windows
    if window_drop_rate > 0:
        # Don't drop transition windows (types 2, 3)
        is_transition = (event_types == 2) | (event_types == 3)
        can_drop = padding_mask & ~is_transition
        drop_mask = torch.rand_like(can_drop.float()) < window_drop_rate
        drop_mask = drop_mask & can_drop
        padding_mask = padding_mask & ~drop_mask

    # 3. Channel dropout — randomly zero out channels
    if channel_drop_rate > 0:
        B, W, C, T = windows.shape
        # Per-sample channel mask (same mask for all windows in a sample)
        channel_mask = torch.rand(B, 1, C, 1, device=windows.device) > channel_drop_rate
        windows = windows * channel_mask.float()

    batch["windows"] = windows
    batch["padding_mask"] = padding_mask
    return batch
