"""EDCC Training Entry Point.

Usage:
    python -m edcc.scripts.run_edcc_train [--config path/to/config.yaml] [--seed 42]
"""

import argparse
import os
import sys
import random

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch.utils.data import DataLoader
from datasets.caueeg_script import load_caueeg_task_datasets
from edcc.data.windowed_dataset import EDCCDataset
from edcc.data.collate import edcc_collate_fn
from edcc.models.edcc_model import EDCCModel
from edcc.training.trainer import EDCCTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="EDCC Training")
    parser.add_argument(
        "--config",
        type=str,
        default="edcc/configs/edcc_base.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    set_seed(seed)
    print(f"Seed: {seed}")

    # Device
    if not torch.cuda.is_available():
        print("WARNING: No GPU available, using CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load CAUEEG dataset (reuse existing code)
    print("Loading CAUEEG dataset...")
    dataset_path = data_cfg["dataset_path"]
    _, train_ds_raw, val_ds_raw, test_ds_raw = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=data_cfg["task"],
        load_event=True,
        file_format=data_cfg["file_format"],
    )
    print(f"  Train: {len(train_ds_raw)}, Val: {len(val_ds_raw)}, Test: {len(test_ds_raw)}")

    # Wrap with EDCCDataset
    train_dataset = EDCCDataset(
        train_ds_raw,
        window_size=data_cfg["window_size"],
        window_stride=data_cfg["window_stride"],
        max_windows=data_cfg["max_windows_train"],
        mode="train",
        transition_margin=data_cfg["transition_margin"],
        normalize=data_cfg["normalize"],
        norm_mode=data_cfg.get("norm_mode", "sample"),
    )
    eval_max_windows = data_cfg.get("max_windows_eval", 128)
    val_dataset = EDCCDataset(
        val_ds_raw,
        window_size=data_cfg["window_size"],
        window_stride=data_cfg["window_stride"],
        max_windows=eval_max_windows,
        mode="eval",
        transition_margin=data_cfg["transition_margin"],
        normalize=data_cfg["normalize"],
        norm_mode=data_cfg.get("norm_mode", "sample"),
    )
    test_dataset = EDCCDataset(
        test_ds_raw,
        window_size=data_cfg["window_size"],
        window_stride=data_cfg["window_stride"],
        max_windows=eval_max_windows,
        mode="eval",
        transition_margin=data_cfg["transition_margin"],
        normalize=data_cfg["normalize"],
        norm_mode=data_cfg.get("norm_mode", "sample"),
    )

    # DataLoaders
    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=edcc_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 4),  # Smaller batch for eval (more windows)
        shuffle=False,
        collate_fn=edcc_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(1, batch_size // 4),
        shuffle=False,
        collate_fn=edcc_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Quick data check
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  windows: {sample_batch['windows'].shape}")
    print(f"  event_types: {sample_batch['event_types'].shape}")
    print(f"  padding_mask: {sample_batch['padding_mask'].shape}")

    # Build model
    model = EDCCModel(**model_cfg)
    total_params, trainable_params = model.count_parameters()
    print(f"\nModel: EDCC")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # TTA dataset: larger window budget + uniform random (no stratified bias)
    tta_max = data_cfg.get("max_windows_eval", 256)
    tta_dataset = EDCCDataset(
        test_ds_raw,
        window_size=data_cfg["window_size"],
        window_stride=data_cfg["window_stride"],
        max_windows=tta_max,
        mode="eval",  # No stratified bias, just uniform subsampling if over limit
        transition_margin=data_cfg["transition_margin"],
        normalize=data_cfg["normalize"],
        norm_mode=data_cfg.get("norm_mode", "sample"),
        preload=False,
    )
    # Share cache with test_dataset to avoid double loading
    if hasattr(test_dataset, '_cache') and test_dataset._cache is not None:
        tta_dataset._cache = test_dataset._cache

    # Train
    trainer = EDCCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=train_cfg,
        device=device,
    )
    trainer.test_dataset_for_tta = tta_dataset

    print(f"\n{'='*60}")
    print(f"Starting EDCC training...")
    print(f"{'='*60}\n")

    trainer.train()


if __name__ == "__main__":
    main()
