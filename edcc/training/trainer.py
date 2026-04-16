"""EDCC Training Loop.

Handles training, validation, checkpointing, and logging.
"""

import os
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from .losses import EDCCLoss
from ..data.augmentation import apply_augmentations, apply_mixup


class EDCCTrainer:
    """Trainer for the EDCC model.

    Args:
        model: EDCCModel instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader (optional).
        config: Training configuration dict.
        device: torch device.
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None, config=None, device=None):
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Optimizer
        lr = self.config.get("lr", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.05)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        total_epochs = self.config.get("total_epochs", 100)
        warmup_epochs = self.config.get("warmup_epochs", 5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=lr * 0.01
        )
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

        # CORAL mode
        self.use_coral = self.config.get("use_coral", False)

        # Loss
        self.criterion = EDCCLoss(
            alpha=self.config.get("loss_alpha", 0.5),
            beta=self.config.get("loss_beta", 0.5),
            gamma=self.config.get("loss_gamma", 0.1),
            class_weights=self.config.get("class_weights", None),
            focal_gamma=self.config.get("focal_gamma", 0.0),
            use_coral=self.use_coral,
            ordinal_weight=self.config.get("ordinal_weight", 0.0),
            use_race=self.config.get("use_race", False),
            race_epsilon=self.config.get("race_epsilon", 0.15),
            race_sigma=self.config.get("race_sigma", 0.8),
            label_smoothing=self.config.get("label_smoothing", 0.1),
        )

        # AMP
        self.use_amp = self.config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Augmentation params
        self.aug_noise_std = self.config.get("aug_noise_std", 0.05)
        self.aug_window_drop = self.config.get("aug_window_drop", 0.1)
        self.aug_channel_drop = self.config.get("aug_channel_drop", 0.1)
        self.mixup_alpha = self.config.get("mixup_alpha", 0.2)

        # Checkpointing
        self.checkpoint_dir = self.config.get("checkpoint_dir", "local/edcc_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_acc = 0.0

        # Early stopping
        self.patience = self.config.get("early_stopping_patience", 15)
        self.patience_counter = 0

    def train(self):
        """Run the full training loop."""
        for epoch in range(1, self.total_epochs + 1):
            # Warmup LR
            if epoch <= self.warmup_epochs:
                warmup_lr = self.config.get("lr", 1e-4) * epoch / self.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Train one epoch
            train_loss, train_acc = self._train_epoch(epoch)

            # Validate
            val_loss, val_acc, val_metrics = self._validate(self.val_loader)

            # LR step (after warmup)
            if epoch > self.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.total_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"LR: {current_lr:.2e}"
            )

            # Checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
                self.patience_counter = 0
                print(f"  -> New best val accuracy: {val_acc:.2f}%")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} (patience={self.patience})")
                break

        # Final test evaluation
        if self.test_loader:
            self._load_best_checkpoint()
            test_loss, test_acc, test_metrics = self._validate(self.test_loader)
            print(f"\n=== Test Results ===")
            print(f"Test Accuracy: {test_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f}")
            for k, v in test_metrics.items():
                print(f"  {k}: {v:.4f}")

            # TTA evaluation
            tta_repeats = self.config.get("tta_repeats", 8)
            if tta_repeats > 1 and hasattr(self, "test_dataset_for_tta"):
                tta_acc, tta_metrics = self._validate_tta(
                    self.test_dataset_for_tta, tta_repeats
                )
                print(f"\n=== Test Results (TTA x{tta_repeats}) ===")
                print(f"TTA Accuracy: {tta_acc:.2f}%")
                for k, v in tta_metrics.items():
                    print(f"  {k}: {v:.4f}")

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            # Move to device
            batch = self._to_device(batch)

            # Apply augmentations
            batch = apply_augmentations(
                batch,
                noise_std=self.aug_noise_std,
                window_drop_rate=self.aug_window_drop,
                channel_drop_rate=self.aug_channel_drop,
            )

            # Apply mixup
            batch, mixup_info = apply_mixup(batch, alpha=self.mixup_alpha)

            self.optimizer.zero_grad()

            with autocast("cuda", enabled=self.use_amp):
                logits, aux = self.model(
                    windows=batch["windows"],
                    event_types=batch["event_types"],
                    positions=batch["positions"],
                    age=batch["age"],
                    padding_mask=batch["padding_mask"],
                )
                age_pred = aux.get("age_pred") if isinstance(aux, dict) else aux
                window_logits = aux.get("window_logits") if isinstance(aux, dict) else None
                window_aux_w = self.config.get("window_aux_weight", 0.3)

                if mixup_info is not None:
                    lam = mixup_info["lam"]
                    loss_a, _ = self.criterion(
                        logits=logits, targets=mixup_info["targets_a"],
                        age_pred=age_pred, age_true=batch["age"],
                        window_logits=window_logits, padding_mask=batch["padding_mask"],
                        window_aux_weight=window_aux_w,
                    )
                    loss_b, _ = self.criterion(
                        logits=logits, targets=mixup_info["targets_b"],
                        age_pred=age_pred, age_true=batch["age"],
                        window_logits=window_logits, padding_mask=batch["padding_mask"],
                        window_aux_weight=window_aux_w,
                    )
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss, _ = self.criterion(
                        logits=logits,
                        targets=batch["class_label"],
                        age_pred=age_pred,
                        age_true=batch["age"],
                        serials=batch["serial"],
                        window_logits=window_logits,
                        padding_mask=batch["padding_mask"],
                        window_aux_weight=window_aux_w,
                    )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch["class_label"].size(0)
            if self.use_coral:
                from ..training.losses import CORALLoss
                preds = CORALLoss.logits_to_pred(logits)
            else:
                preds = logits.argmax(dim=1)
            correct += (preds == batch["class_label"]).sum().item()
            total += batch["class_label"].size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self, loader):
        """Validate on a data loader."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            batch = self._to_device(batch)

            logits, _aux = self.model(
                windows=batch["windows"],
                event_types=batch["event_types"],
                positions=batch["positions"],
                age=batch["age"],
                padding_mask=batch["padding_mask"],
            )

            loss, _ = self.criterion(logits, batch["class_label"])

            total_loss += loss.item() * batch["class_label"].size(0)
            if self.use_coral:
                from ..training.losses import CORALLoss
                preds = CORALLoss.logits_to_pred(logits)
            else:
                preds = logits.argmax(dim=1)
            correct += (preds == batch["class_label"]).sum().item()
            total += batch["class_label"].size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch["class_label"].cpu().tolist())

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        # Per-class metrics
        metrics = self._compute_class_metrics(all_preds, all_targets)

        return avg_loss, accuracy, metrics

    @torch.no_grad()
    def _validate_tta(self, dataset, n_repeats=8):
        """Test-Time Augmentation with diverse window offsets.

        Each repeat uses a different temporal offset for window extraction,
        providing genuinely different views of each recording.
        """
        self.model.eval()
        import numpy as np
        from ..data.windowed_dataset import EDCCDataset
        from ..data.collate import edcc_collate_fn

        serial_logits = {}
        serial_targets = {}

        base_stride = dataset.window_stride  # e.g., 200

        for repeat in range(n_repeats):
            # Shift window positions by different offsets each repeat
            if repeat == 0:
                offset = 0
            else:
                offset = int(base_stride * repeat / n_repeats)

            # Create a modified dataset with shifted windows
            shifted_ds = _OffsetDatasetWrapper(dataset, offset=offset)

            loader = torch.utils.data.DataLoader(
                shifted_ds, batch_size=1, shuffle=False,
                collate_fn=edcc_collate_fn, num_workers=0,
            )

            for batch in loader:
                batch = self._to_device(batch)

                logits, _aux = self.model(
                    windows=batch["windows"],
                    event_types=batch["event_types"],
                    positions=batch["positions"],
                    age=batch["age"],
                    padding_mask=batch["padding_mask"],
                )

                for i, serial in enumerate(batch["serial"]):
                    if serial not in serial_logits:
                        serial_logits[serial] = []
                        serial_targets[serial] = batch["class_label"][i].item()
                    serial_logits[serial].append(logits[i].cpu())

        # Average logits and predict
        all_preds = []
        all_targets = []
        correct = 0
        total = 0

        for serial in serial_logits:
            avg_logit = torch.stack(serial_logits[serial]).mean(dim=0)
            pred = avg_logit.argmax().item()
            target = serial_targets[serial]
            all_preds.append(pred)
            all_targets.append(target)
            correct += int(pred == target)
            total += 1

        accuracy = 100.0 * correct / max(total, 1)
        metrics = self._compute_class_metrics(all_preds, all_targets)
        return accuracy, metrics

    def _compute_class_metrics(self, preds, targets):
        """Compute per-class sensitivity, specificity, balanced accuracy, macro F1."""
        import numpy as np
        preds = np.array(preds)
        targets = np.array(targets)
        num_classes = 3
        class_names = ["Normal", "MCI", "Dementia"]
        metrics = {}
        sensitivities = []
        f1_scores = []

        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum()
            fn = ((preds != c) & (targets == c)).sum()
            fp = ((preds == c) & (targets != c)).sum()
            tn = ((preds != c) & (targets != c)).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

            metrics[f"{class_names[c]}_sensitivity"] = sensitivity
            metrics[f"{class_names[c]}_specificity"] = specificity
            metrics[f"{class_names[c]}_f1"] = f1
            sensitivities.append(sensitivity)
            f1_scores.append(f1)

        metrics["balanced_accuracy"] = np.mean(sensitivities)
        metrics["macro_f1"] = np.mean(f1_scores)

        return metrics

    def _to_device(self, batch):
        """Move batch tensors to device."""
        for key in ["windows", "event_types", "positions", "padding_mask",
                     "num_windows", "age", "class_label"]:
            if key in batch and torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        return batch

    def _save_checkpoint(self, epoch, val_acc):
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
        }, path)

    def _load_best_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.2f}%)")


class _OffsetDatasetWrapper(torch.utils.data.Dataset):
    """Wraps an EDCCDataset to shift window extraction by a fixed offset.

    This provides genuinely different windows for each TTA repeat.
    """

    def __init__(self, base_dataset, offset=0):
        self.base = base_dataset
        self.offset = offset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        import numpy as np
        from ..data.event_segmenter import parse_events, get_window_event_type

        if self.base._cache is not None:
            cached = self.base._cache[idx]
            eeg = cached["signal"]
            events = cached["event"]
            age = cached["age"]
            class_label = cached["class_label"]
            serial = cached["serial"]
        else:
            sample = self.base.base_dataset[idx]
            from ..data.windowed_dataset import EEG_CHANNEL_INDICES
            eeg = sample["signal"][EEG_CHANNEL_INDICES]
            events = sample.get("event", [])
            age = sample["age"]
            class_label = sample["class_label"]
            serial = sample["serial"]

        signal_length = eeg.shape[1]
        segments = parse_events(events, signal_length, transition_margin=self.base.transition_margin)

        ws = self.base.window_size
        stride = self.base.window_stride

        # Apply offset to window start positions
        windows = []
        event_types = []
        positions = []

        for start in range(self.offset, signal_length - ws + 1, stride):
            end = start + ws
            w = eeg[:, start:end]
            if w.shape[1] < ws or np.any(np.isnan(w)) or np.std(w) < 1e-10:
                continue
            etype = get_window_event_type(segments, start, end)
            windows.append(w)
            event_types.append(etype)
            positions.append(start + ws // 2)

        if not windows:
            w = eeg[:, :ws]
            if w.shape[1] < ws:
                w = np.pad(w, ((0, 0), (0, ws - w.shape[1])))
            windows = [w]
            event_types = [5]
            positions = [0]

        # Limit windows
        if self.base.max_windows and len(windows) > self.base.max_windows:
            indices = np.random.choice(len(windows), self.base.max_windows, replace=False)
            indices.sort()
            windows = [windows[i] for i in indices]
            event_types = [event_types[i] for i in indices]
            positions = [positions[i] for i in indices]

        stacked = np.stack(windows, axis=0)
        if self.base.normalize:
            mean = np.mean(stacked, axis=(0, 2), keepdims=True)
            std = np.std(stacked, axis=(0, 2), keepdims=True) + 1e-8
            stacked = (stacked - mean) / std

        import torch as th
        pos_t = th.tensor(positions, dtype=th.float32)
        if signal_length > 0:
            pos_t = pos_t / signal_length

        return {
            "windows": th.from_numpy(stacked).float(),
            "event_types": th.tensor(event_types, dtype=th.long),
            "positions": pos_t,
            "num_windows": len(windows),
            "age": th.tensor(age, dtype=th.float32),
            "class_label": th.tensor(class_label, dtype=th.long),
            "serial": serial,
        }
