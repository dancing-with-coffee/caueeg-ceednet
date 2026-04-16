"""Custom collate function for variable-length window sequences."""

import torch


def edcc_collate_fn(batch):
    """Collate function that pads window sequences to the max length in the batch.

    Args:
        batch: List of dicts from EDCCDataset.__getitem__().

    Returns:
        Dict with padded tensors:
            windows: (B, max_W, 19, window_size)
            event_types: (B, max_W)
            positions: (B, max_W)
            padding_mask: (B, max_W) — True for valid (non-padded) positions
            num_windows: (B,)
            age: (B,)
            class_label: (B,)
            serial: list of strings
    """
    max_w = max(item["num_windows"] for item in batch)
    batch_size = len(batch)
    n_channels = batch[0]["windows"].shape[1]
    window_size = batch[0]["windows"].shape[2]

    # Pre-allocate padded tensors
    windows = torch.zeros(batch_size, max_w, n_channels, window_size)
    event_types = torch.full((batch_size, max_w), fill_value=5, dtype=torch.long)  # OTHER
    positions = torch.zeros(batch_size, max_w)
    padding_mask = torch.zeros(batch_size, max_w, dtype=torch.bool)

    ages = []
    class_labels = []
    serials = []
    num_windows = []

    for i, item in enumerate(batch):
        w = item["num_windows"]
        windows[i, :w] = item["windows"]
        event_types[i, :w] = item["event_types"]
        positions[i, :w] = item["positions"]
        padding_mask[i, :w] = True

        ages.append(item["age"])
        class_labels.append(item["class_label"])
        serials.append(item["serial"])
        num_windows.append(w)

    return {
        "windows": windows,
        "event_types": event_types,
        "positions": positions,
        "padding_mask": padding_mask,
        "num_windows": torch.tensor(num_windows, dtype=torch.long),
        "age": torch.stack(ages),
        "class_label": torch.stack(class_labels),
        "serial": serials,
    }
