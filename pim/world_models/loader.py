"""Checkpoint and dataset loaders — model-agnostic where possible.

load_checkpoint dispatches on the keys in ckpt["model_config"]: it picks
GRUModel if the config has 'num_layers', RSSMModel if it has 'det_size',
DiTModel if it has 'n_sample_steps'.

load_dataset reads a directory containing dataset.json + test.h5 + edits.h5
(and train.h5 + val.h5; these are not needed for eval but are part of the
shared layout). Returns numpy arrays + config baselines convenient for
eval/figures consumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from pim.simulator.dataset import reconstruct_clean_obs
from pim.world_models.dataloader import ObservationDataset
from pim.world_models.dit import DiTModel
from pim.world_models.dit import ModelConfig as DiTConfig
from pim.world_models.gru import GRUModel
from pim.world_models.gru import ModelConfig as GRUConfig
from pim.world_models.protocol import HiddenStateModel
from pim.world_models.rssm import ModelConfig as RSSMConfig
from pim.world_models.rssm import RSSMModel


@dataclass
class CheckpointInfo:
    """Lightweight bundle of training-time metadata."""
    epoch: int
    val_loss: float
    model_config: dict
    train_config: dict
    metrics_history: list[dict]
    run_name: str


@dataclass
class Dataset:
    """Pre-loaded arrays + config for one evaluation split.

    Attributes
    ----------
    obs            : (N, T, R) noisy observations
    clean_obs      : (N, T, R) noiseless observations (reconstructed)
    positions      : (N, T, max_obj, 2)
    is_visible     : (N, T, max_obj) bool
    obs_id         : (N, T, R) int8
    reflectivities : (N, max_obj)
    config         : full dataset config dict (from dataset.json or HDF5 attrs)
    obs_noise_std  : extracted from config
    position_noise_std : extracted from config
    h5_path        : path to the source HDF5 (used by lazy plot helpers)
    T_frames, obs_res : convenience scalars
    """
    obs: np.ndarray
    clean_obs: np.ndarray
    positions: np.ndarray
    is_visible: np.ndarray
    obs_id: np.ndarray
    reflectivities: np.ndarray
    config: dict
    obs_noise_std: float
    position_noise_std: float
    h5_path: str
    T_frames: int
    obs_res: int

    @property
    def n_samples(self) -> int:
        return self.obs.shape[0]


@dataclass
class EditsData:
    """Edits dataset — observations + GT positions + edit metadata."""
    obs: np.ndarray              # (N, T, R)
    clean_obs: np.ndarray        # (N, T, R)
    positions: np.ndarray        # (N, T, max_obj, 2)
    colors: np.ndarray           # (N, max_obj, 3)
    edit_frame: int              # uniform across samples (taken from row 0)
    edit_object: np.ndarray      # (N,)
    edit_op: np.ndarray          # (N,) byte strings
    edit_value: np.ndarray       # (N, 2)
    h5_path: str
    T_frames: int
    obs_res: int

    @property
    def n_samples(self) -> int:
        return self.obs.shape[0]


# ── Model checkpoint ──────────────────────────────────────────────────────────


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[HiddenStateModel, CheckpointInfo]:
    """Load a checkpoint; dispatch on model_config keys to pick GRU vs RSSM.

    Returns
    -------
    model : HiddenStateModel in eval mode with no grad
    info  : CheckpointInfo
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    mcfg_dict = ckpt["model_config"]

    if "det_size" in mcfg_dict:
        model: HiddenStateModel = RSSMModel(RSSMConfig(**mcfg_dict)).to(device)
    elif "n_sample_steps" in mcfg_dict:
        model = DiTModel(DiTConfig(**mcfg_dict)).to(device)
    elif "num_layers" in mcfg_dict or "hidden_size" in mcfg_dict:
        model = GRUModel(GRUConfig(**mcfg_dict)).to(device)
    else:
        raise ValueError(
            f"Cannot infer model type from model_config keys: {list(mcfg_dict)}"
        )

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    metrics_path = Path(checkpoint_path).parent / "metrics.jsonl"
    metrics_history = (
        [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if metrics_path.exists() else []
    )

    info = CheckpointInfo(
        epoch=ckpt["epoch"],
        val_loss=ckpt["val_loss"],
        model_config=mcfg_dict,
        train_config=ckpt.get("train_config", {}),
        metrics_history=metrics_history,
        run_name=Path(checkpoint_path).parent.name,
    )
    return model, info


# ── Dataset ───────────────────────────────────────────────────────────────────


def _load_h5_dataset(
    h5_path: str | Path,
    *,
    n_obj_keep: int | None = None,
) -> Dataset:
    """Load test/val split: obs + positions + is_visible + clean_obs + config."""
    with h5py.File(h5_path, "r") as f:
        T = f["obs_intensity"].shape[1]
        R = f["obs_intensity"].shape[2]
        obs = f["obs_intensity"][:].astype(np.float32)
        max_obj = f["positions"].shape[2]
        n_obj = n_obj_keep if n_obj_keep is not None else max_obj
        positions = f["positions"][:, :, :n_obj, :].astype(np.float32)
        is_visible = f["is_visible"][:, :, :n_obj].astype(bool)
        obs_id = f["obs_id"][:].astype(np.int8)
        reflectivities = f["reflectivities"][:].astype(np.float32)
        config = json.loads(f.attrs["config_json"])

    clean_obs = reconstruct_clean_obs(obs_id, reflectivities)
    return Dataset(
        obs=obs,
        clean_obs=clean_obs,
        positions=positions,
        is_visible=is_visible,
        obs_id=obs_id,
        reflectivities=reflectivities,
        config=config,
        obs_noise_std=float(config["dataset"]["sim"]["obs_noise_std"]),
        position_noise_std=float(config["dataset"]["sim"]["position_noise_std"]),
        h5_path=str(h5_path),
        T_frames=T,
        obs_res=R,
    )


def _load_edits(
    h5_path: str | Path,
    *,
    n_obj_keep: int | None = None,
) -> EditsData:
    """Load edits split with intervention metadata."""
    with h5py.File(h5_path, "r") as f:
        T = f["obs_intensity"].shape[1]
        R = f["obs_intensity"].shape[2]
        obs = f["obs_intensity"][:].astype(np.float32)
        max_obj = f["positions"].shape[2]
        n_obj = n_obj_keep if n_obj_keep is not None else max_obj
        positions = f["positions"][:, :, :n_obj, :].astype(np.float32)
        colors = f["colors"][:, :n_obj, :].astype(np.float32)
        obs_id = f["obs_id"][:].astype(np.int8)
        reflectivities = f["reflectivities"][:].astype(np.float32)
        edit_frame = int(f["edit_frame"][0])
        edit_object = f["edit_object"][:]
        edit_op = f["edit_op"][:]
        edit_value = f["edit_value"][:]

    clean_obs = reconstruct_clean_obs(obs_id, reflectivities)
    return EditsData(
        obs=obs,
        clean_obs=clean_obs,
        positions=positions,
        colors=colors,
        edit_frame=edit_frame,
        edit_object=edit_object,
        edit_op=edit_op,
        edit_value=edit_value,
        h5_path=str(h5_path),
        T_frames=T,
        obs_res=R,
    )


@dataclass
class DatasetBundle:
    """A dataset directory loaded for evaluation: test split + optional edits."""
    data_dir: Path
    test: Dataset
    edits: EditsData | None


def load_dataset(
    data_dir: str | Path,
    *,
    n_obj_keep: int | None = None,
    require_edits: bool = True,
) -> DatasetBundle:
    """Load a dataset directory: dataset.json + test.h5 + edits.h5.

    Parameters
    ----------
    data_dir     : directory containing dataset.json, train.h5, val.h5, test.h5, edits.h5
    n_obj_keep   : keep only this many objects along the object axis (for probes
                   that assume a fixed n_obj). None keeps max_obj.
    require_edits : if True and edits.h5 is missing, raise.

    Returns
    -------
    DatasetBundle with .test and .edits (or .edits=None if require_edits=False
    and missing)
    """
    d = Path(data_dir)
    test_path = d / "test.h5"
    edits_path = d / "edits.h5"

    if not test_path.exists():
        raise FileNotFoundError(f"test split not found at {test_path}")

    test = _load_h5_dataset(test_path, n_obj_keep=n_obj_keep)

    edits: EditsData | None = None
    if edits_path.exists():
        edits = _load_edits(edits_path, n_obj_keep=n_obj_keep)
    elif require_edits:
        raise FileNotFoundError(f"edits split not found at {edits_path}")

    return DatasetBundle(data_dir=d, test=test, edits=edits)


def make_test_loader(
    dataset: Dataset,
    *,
    batch_size: int = 512,
    num_workers: int = 0,
    keys: tuple[str, ...] = ("obs_intensity", "positions", "is_visible"),
) -> DataLoader:
    """Build a DataLoader over the full test set (no shuffling)."""
    ds = ObservationDataset(dataset.h5_path, np.arange(dataset.n_samples), keys=keys)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=(num_workers > 0), persistent_workers=(num_workers > 0),
    )
