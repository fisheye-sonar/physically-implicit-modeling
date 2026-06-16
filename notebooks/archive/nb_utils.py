"""Shared utilities for GRU evaluation / probe notebooks.

Thin helpers for model + loader construction only.
Inference utilities (teacher forcing, AR rollout) are in pim.eval._helpers.
Dataset loading (load_sample) is in pim.simulator.dataset.
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pim.world_models.gru import GRUModel, ModelConfig
from pim.world_models.dataloader import ObservationDataset


def build_loader(
    h5_path: str,
    indices: np.ndarray | None = None,
    keys: tuple[str, ...] = ("obs_intensity",),
    batch_size: int = 512,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader over given indices (all samples if None)."""
    with h5py.File(h5_path, "r") as f:
        n = f["obs_intensity"].shape[0]
    if indices is None:
        indices = np.arange(n)
    ds = ObservationDataset(h5_path, indices, keys=keys)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[GRUModel, dict]:
    """Load a GRU checkpoint.

    Returns
    -------
    model : GRUModel in eval mode with no grad
    info  : dict with keys epoch, val_loss, model_config, train_config
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = GRUModel(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
        "model_config": ckpt["model_config"],
        "train_config": ckpt["train_config"],
    }
