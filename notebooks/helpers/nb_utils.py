"""Shared utilities for GRU evaluation / probe notebooks."""
from __future__ import annotations

import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pim.config import SimConfig
from pim.sim import Scene
from pim.models.gru import GRUModel, ModelConfig
from pim.models.dataloader import ObservationDataset


# ── Dataset helpers ───────────────────────────────────────────────────────────


def load_sample(
    path: str, idx: int
) -> tuple[Scene, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a Scene and stored observations from one HDF5 row."""
    with h5py.File(path, "r") as f:
        cfg = SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])
        n = int(f["n_objects"][idx])
        positions      = f["positions"][idx, :, :n, :].astype(np.float64)   # (T, n, 2)
        velocities     = f["velocities"][idx, :, :n, :].astype(np.float64)  # (T, n, 2)
        colors         = f["colors"][idx, :n, :].astype(np.float64)          # (n, 3)
        reflectivities = f["reflectivities"][idx, :n].astype(np.float64)     # (n,)
        radii          = f["radii"][idx, :n].astype(np.float64)              # (n,)
        obs_depth      = f["obs_depth"][idx].astype(np.float32)             # (T, R)
        obs_id         = f["obs_id"][idx]                                    # (T, R)
        obs_intensity  = f["obs_intensity"][idx].astype(np.float32)         # (T, R)
    scene = Scene(
        positions=positions, velocities=velocities, radii=radii,
        colors=colors, reflectivities=reflectivities, config=cfg,
    )
    return scene, obs_depth, obs_id, obs_intensity


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


# ── Model helpers ─────────────────────────────────────────────────────────────


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


@torch.no_grad()
def get_hidden_states(
    model: GRUModel,
    obs: torch.Tensor | np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """Run teacher-forcing forward; return per-timestep hidden states.

    Parameters
    ----------
    obs : (B, T, R) tensor or ndarray

    Returns
    -------
    h : (B, T-1, hidden_size) tensor
        h[:, t, :] is the hidden state produced after the model sees obs[:, t, :].
        Aligns with positions[:, t, :] and is_visible[:, t, :].
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).float()
    obs = obs.to(device)
    x = F.relu(model.encoder(obs[:, :-1, :]))  # (B, T-1, H)
    h, _ = model.gru(x)                         # (B, T-1, H)
    return h


@torch.no_grad()
def autoregressive_rollout(
    model: GRUModel,
    obs_np: np.ndarray,
    n_context: int,
    device: str = "cpu",
) -> np.ndarray:
    """Warm up on obs_np[:n_context], then roll out autoregressively.

    Parameters
    ----------
    obs_np    : (T, R) float32 ndarray
    n_context : number of real observations used to build hidden state

    Returns
    -------
    pred : (T - n_context, R) float32 ndarray
        Predicted observations for frames n_context .. T-1.
    """
    T = obs_np.shape[0]
    obs_t = torch.from_numpy(obs_np).float().to(device)
    h = None
    for t in range(n_context):
        pred_t, h = model.step(obs_t[t].unsqueeze(0), h)
    preds = []
    x = pred_t
    for _ in range(T - n_context):
        x, h = model.step(x, h)
        preds.append(x.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)
