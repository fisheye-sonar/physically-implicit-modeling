"""Architecture-agnostic dataloader for HDF5 observation datasets.

Each sample is a dict of FloatTensors keyed by HDF5 dataset name.  The
default key is ``obs_intensity`` which yields a ``(T, R)`` tensor.  Other
keys (e.g. ``obs_depth``) can be requested at construction time.

Usage
-----
    train_loader, val_loader = build_dataloaders(
        "datasets/initial_easy_100k/dataset.h5",
        batch_size=256,
        num_workers=4,
    )
    for batch in train_loader:
        obs = batch["obs_intensity"]  # (B, T, R)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ObservationDataset(Dataset):
    """Lazily loads named arrays from an HDF5 file.

    The HDF5 file is opened once per worker process (on first ``__getitem__``
    call) and kept open for the lifetime of that worker.  This is safe with
    PyTorch's ``persistent_workers=True`` and avoids the overhead of
    opening/closing the file on every sample.

    Parameters
    ----------
    h5_path:
        Path to the HDF5 file produced by ``generate_dataset``.
    indices:
        Sample indices to include (e.g. train split or val split).
    keys:
        HDF5 dataset names to load.  Each becomes a key in the returned dict.
        Shapes must be ``(N, ...)`` so that indexing by sample index works.
    """

    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray,
        keys: Sequence[str] = ("obs_intensity",),
    ) -> None:
        self.h5_path = str(h5_path)
        self.indices = indices
        self.keys = list(keys)
        self._file: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        f = self._open()
        return {
            key: torch.from_numpy(f[key][idx].astype(np.float32))
            for key in self.keys
        }


def build_dataloaders(
    h5_path: str | Path,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
    num_workers: int = 4,
    keys: Sequence[str] = ("obs_intensity",),
) -> tuple[DataLoader, DataLoader]:
    """Split samples into train/val and return a DataLoader for each.

    Parameters
    ----------
    h5_path:
        Path to the HDF5 dataset file.
    val_fraction:
        Fraction of samples to hold out for validation.
    batch_size:
        Batch size for both loaders.
    seed:
        RNG seed for the train/val split (reproducible).
    num_workers:
        Number of DataLoader worker processes.  Use 0 for debugging.
    keys:
        HDF5 keys to include in each batch.

    Returns
    -------
    train_loader, val_loader
    """
    with h5py.File(h5_path, "r") as f:
        n_samples = f["obs_intensity"].shape[0]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = ObservationDataset(h5_path, train_idx, keys=keys)
    val_ds = ObservationDataset(h5_path, val_idx, keys=keys)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
