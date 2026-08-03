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


class InMemoryObservationDataset(Dataset):
    """Loads named arrays fully into RAM as torch tensors.

    The HDF5 datasets are gzip-compressed; random per-sample access pays the
    decompression cost on every read (~30× slower than the model step for
    training-sized batches).  For datasets that fit in RAM (~2 GB for 90k
    samples) this loads each key once and serves views, removing the
    bottleneck.  Use ``num_workers=0`` — there is nothing to parallelise.
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
        with h5py.File(self.h5_path, "r") as f:
            # Sort for h5py fancy indexing; keep the caller's order after.
            order = np.argsort(indices)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            self._data = {
                key: torch.from_numpy(
                    f[key][np.asarray(indices)[order]].astype(np.float32)
                )[inverse]
                for key in self.keys
            }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {key: self._data[key][i] for key in self.keys}


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
            key: torch.from_numpy(f[key][idx].astype(np.float32)) for key in self.keys
        }


class InMemoryLoader:
    """Device-resident batch iterator over one pre-loaded array.

    Reading this repo's gzip-compressed HDF5 sample-by-sample makes GRU training
    CPU-bound (measured: 68 s/epoch at hidden 256 on dataset 4, with the GPU
    mostly idle).  The observation tensor is small enough to live on the GPU
    outright (90k x 40 x 128 float32 = 1.8 GB), so a whole epoch becomes pure
    device compute.

    Semantics are deliberately identical to the `DataLoader` path it replaces:
    the same train/val index split, the same batch size, reshuffled every epoch
    for train and fixed order for val, no dropping of the last partial batch.
    Yields the same `{key: tensor}` batch dicts, already on `device`.
    """

    def __init__(
        self,
        data: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        key: str = "obs_intensity",
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self) -> int:
        return (self.data.shape[0] + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.data.shape[0]
        order = (
            torch.randperm(n, generator=self._gen) if self.shuffle else torch.arange(n)
        )
        order = order.to(self.data.device)
        for i in range(0, n, self.batch_size):
            yield {self.key: self.data[order[i : i + self.batch_size]]}


def build_inmemory_dataloaders(
    h5_path: str | Path,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
    device: str | torch.device = "cuda",
    key: str = "obs_intensity",
) -> tuple[InMemoryLoader, InMemoryLoader]:
    """Same split/batching contract as `build_dataloaders`, fully device-resident.

    Reads `key` from the HDF5 file once, moves it to `device`, and returns two
    `InMemoryLoader`s.  The train/val split uses the identical RNG call as
    `build_dataloaders`, so a run is comparable to one trained through the
    lazy loader.
    """
    with h5py.File(h5_path, "r") as f:
        arr = f[key][:].astype(np.float32)

    n_samples = arr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    tensor = torch.from_numpy(arr).to(device)
    return (
        InMemoryLoader(
            tensor[train_idx], batch_size=batch_size, shuffle=True, seed=seed, key=key
        ),
        InMemoryLoader(
            tensor[val_idx], batch_size=batch_size, shuffle=False, seed=seed, key=key
        ),
    )


def build_dataloaders(
    h5_path: str | Path,
    val_fraction: float = 0.1,
    batch_size: int = 256,
    seed: int = 0,
    num_workers: int = 4,
    keys: Sequence[str] = ("obs_intensity",),
    in_memory: bool = False,
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
        Ignored (forced to 0) when ``in_memory=True``.
    keys:
        HDF5 keys to include in each batch.
    in_memory:
        Load the requested keys fully into RAM (see
        InMemoryObservationDataset).  Much faster when the data fits.

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

    ds_cls = InMemoryObservationDataset if in_memory else ObservationDataset
    train_ds = ds_cls(h5_path, train_idx, keys=keys)
    val_ds = ds_cls(h5_path, val_idx, keys=keys)

    if in_memory:
        num_workers = 0
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader
