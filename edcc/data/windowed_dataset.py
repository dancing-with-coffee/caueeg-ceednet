"""Windowed EEG Dataset for EDCC model.

Wraps the existing CauEegDataset to produce fixed-size windows with
event type labels instead of single random crops.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .event_segmenter import parse_events, get_window_event_type, EventType


# 19 EEG channel indices (drop EKG=19, Photic=20)
EEG_CHANNEL_INDICES = list(range(19))

# Brain region groupings (channel index -> region)
BRAIN_REGIONS = {
    "Frontal":  [0, 5, 1, 6, 10, 13, 16],   # Fp1,Fp2,F3,F4,F7,F8,FZ
    "Central":  [2, 7, 17],                    # C3,C4,CZ
    "Parietal": [3, 8, 18],                    # P3,P4,PZ
    "Temporal": [11, 14, 12, 15],              # T3,T4,T5,T6
    "Occipital": [4, 9],                       # O1,O2
}


class EDCCDataset(Dataset):
    """PyTorch Dataset that produces windowed EEG sequences for EDCC.

    Each sample returns all (or a subset of) 2-second windows from a recording,
    along with per-window event type labels.

    Args:
        base_dataset: CauEegDataset instance (with load_event=True).
        window_size: Window size in samples (default: 400 = 2s at 200Hz).
        window_stride: Stride in samples (default: 200 = 1s).
        max_windows: Maximum number of windows per recording during training.
            If None, use all windows. Recommended: 60-80 for training.
        mode: 'train' for random window subsampling, 'eval' for all windows.
        transition_margin: Samples ±around EO↔EC boundaries for transition labels.
        normalize: Per-sample z-score normalization of EEG signals.
    """

    def __init__(
        self,
        base_dataset,
        window_size: int = 400,
        window_stride: int = 200,
        max_windows: int = 64,
        mode: str = "train",
        transition_margin: int = 600,
        normalize: bool = True,
        norm_mode: str = "sample",
        preload: bool = True,
    ):
        self.base_dataset = base_dataset
        self.window_size = window_size
        self.window_stride = window_stride
        self.max_windows = max_windows
        self.mode = mode
        self.transition_margin = transition_margin
        self.norm_mode = norm_mode  # "sample" or "dataset"
        self.normalize = normalize

        # Preload all signals into memory to avoid repeated EDF reads
        self._cache = None
        if preload:
            self._preload()

    def _preload(self):
        """Load all signals and events into memory once."""
        print(f"  Preloading {len(self.base_dataset)} recordings into memory...")
        self._cache = []
        all_means = []
        all_stds = []
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            sig = sample["signal"][EEG_CHANNEL_INDICES].copy()
            self._cache.append({
                "signal": sig,
                "event": sample.get("event", []),
                "age": sample["age"],
                "class_label": sample["class_label"],
                "serial": sample["serial"],
            })
            # Collect per-channel statistics for dataset-level normalization
            if self.norm_mode == "dataset":
                ch_mean = np.mean(sig, axis=1, keepdims=True)  # (19, 1)
                ch_std = np.std(sig, axis=1, keepdims=True)
                all_means.append(ch_mean)
                all_stds.append(ch_std)
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(self.base_dataset)} loaded")

        # Compute dataset-level statistics
        if self.norm_mode == "dataset" and all_means:
            self._dataset_mean = np.mean(all_means, axis=0)  # (19, 1)
            self._dataset_std = np.mean(all_stds, axis=0) + 1e-8
            print(f"  Dataset norm: mean shape={self._dataset_mean.shape}, "
                  f"std range=[{self._dataset_std.min():.2f}, {self._dataset_std.max():.2f}]")

        print(f"  Preloading done.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        if self._cache is not None:
            cached = self._cache[idx]
            eeg = cached["signal"]
            events = cached["event"]
            age = cached["age"]
            class_label = cached["class_label"]
            serial = cached["serial"]
        else:
            sample = self.base_dataset[idx]
            eeg = sample["signal"][EEG_CHANNEL_INDICES]
            events = sample.get("event", [])
            age = sample["age"]
            class_label = sample["class_label"]
            serial = sample["serial"]

        signal_length = eeg.shape[1]

        # Parse events into segments
        segments = parse_events(
            events, signal_length, transition_margin=self.transition_margin
        )

        # Generate all windows
        all_windows, all_event_types, all_positions = self._extract_windows(
            eeg, segments, signal_length
        )

        if len(all_windows) == 0:
            # Fallback: single window from recording start
            w = eeg[:, :self.window_size]
            if w.shape[1] < self.window_size:
                w = np.pad(w, ((0, 0), (0, self.window_size - w.shape[1])))
            all_windows = [w]
            all_event_types = [EventType.OTHER]
            all_positions = [0]

        # Subsample windows for training
        if self.mode == "train" and self.max_windows and len(all_windows) > self.max_windows:
            indices = self._stratified_sample(all_event_types, self.max_windows)
            all_windows = [all_windows[i] for i in indices]
            all_event_types = [all_event_types[i] for i in indices]
            all_positions = [all_positions[i] for i in indices]

        # Stack into tensors
        windows = np.stack(all_windows, axis=0)  # (W, 19, window_size)

        # Normalization
        if self.normalize:
            if self.norm_mode == "dataset" and hasattr(self, "_dataset_mean"):
                # Dataset-level: use precomputed statistics (CeedNet style)
                windows = (windows - self._dataset_mean[np.newaxis]) / self._dataset_std[np.newaxis]
            else:
                # Per-sample: z-score across all windows
                mean = np.mean(windows, axis=(0, 2), keepdims=True)
                std = np.std(windows, axis=(0, 2), keepdims=True) + 1e-8
                windows = (windows - mean) / std

        windows = torch.from_numpy(windows).float()
        event_types = torch.tensor(all_event_types, dtype=torch.long)
        positions = torch.tensor(all_positions, dtype=torch.float32)

        # Normalize positions to [0, 1]
        if signal_length > 0:
            positions = positions / signal_length

        return {
            "windows": windows,                 # (W, 19, window_size)
            "event_types": event_types,          # (W,)
            "positions": positions,              # (W,) normalized temporal positions
            "num_windows": len(all_windows),
            "age": torch.tensor(age, dtype=torch.float32),
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "serial": serial,
        }

    def _extract_windows(self, eeg, segments, signal_length):
        """Extract all valid windows from the EEG signal."""
        windows = []
        event_types = []
        positions = []

        for start in range(0, signal_length - self.window_size + 1, self.window_stride):
            end = start + self.window_size
            window = eeg[:, start:end]

            if window.shape[1] < self.window_size:
                continue

            # Check for NaN or flat signal
            if np.any(np.isnan(window)) or np.std(window) < 1e-10:
                continue

            etype = get_window_event_type(segments, start, end)

            windows.append(window)
            event_types.append(etype)
            positions.append(start + self.window_size // 2)

        return windows, event_types, positions

    def _stratified_sample(self, event_types, n_samples):
        """Sample windows ensuring representation of transition events.

        Transition windows (EO_TO_EC, EC_TO_EO) are always included.
        Remaining slots are filled randomly from other types.
        """
        transition_types = {EventType.EO_TO_EC, EventType.EC_TO_EO}

        transition_indices = [
            i for i, et in enumerate(event_types) if et in transition_types
        ]
        other_indices = [
            i for i, et in enumerate(event_types) if et not in transition_types
        ]

        # Always include all transition windows (up to half of budget)
        max_transition = min(len(transition_indices), n_samples // 2)
        if max_transition < len(transition_indices):
            selected_trans = list(
                np.random.choice(transition_indices, max_transition, replace=False)
            )
        else:
            selected_trans = transition_indices

        # Fill remaining with other windows
        remaining = n_samples - len(selected_trans)
        if remaining > 0 and other_indices:
            n_other = min(remaining, len(other_indices))
            selected_other = list(
                np.random.choice(other_indices, n_other, replace=False)
            )
        else:
            selected_other = []

        indices = sorted(selected_trans + selected_other)
        return indices
