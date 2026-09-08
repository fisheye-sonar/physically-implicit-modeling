"""The discretised state target for discworld — a control against Othello (2026-09-08).

Othello's probe target is categorical (64 tiles × 3 classes); discworld's is continuous
(positions, regressed). This module defines the discworld state as a GRID of cells, each
labelled {0 empty, 1 centre of object 0 (bright), 2 centre of object 1 (dim)} — the same
shape of target as the Othello board probe — so the last nameable difference on the
STATE side of the comparison can be tested with new probes on the same trained model.

The grid lives in the frustum basis the canonical probes already use: lateral ray
coordinate u = x / (scale·y) in [-1, 1] and inverse depth 1/y in [1/y_far, 1/y_near],
binned UNIFORMLY there. That makes every cell the same size in the observation's own
coordinates (rays laterally, apparent width in depth — `frustum.py` explains why depth
enters the frame only as 1/y), and therefore larger at the back than the front in world
space. The observation-exact partition ("movement inside a cell changes no ray") would
need ~128 × 16 ≈ 2,000 cells; NU × ND = 16 × 8 = 128 is the coarsening that keeps the
target at Othello's scale (384 logits vs 192).

The grid covers the REACHABLE region: with `always_in_frustum` a disc centre keeps a
margin of one radius from every wall, so depth runs over [y_near + r, y_far − r] and the
lateral coordinate is normalised by its reach at that depth, u' = u / (1 − r/(scale·y)).
Without this the whole nearest depth row and the outer lateral columns at near depths
are cells nothing ever visits (29 of 128 on the eval split).

Two objects can share a cell only at the far plane (depth bins there span ~3 world
units); the NEARER object's label wins, since it is the one the observation shows.
`label_frames` reports how often that happens.
"""
from __future__ import annotations

import numpy as np

import os

from pim.environments.discworld.frustum import lateral

# Resolution from the GRID environment variable ("<lateral>x<depth>", default 16x8), so a
# coarser arm (8x4: every cell ~4x larger, closer to the model's positional precision)
# runs through the same scripts; outputs of a non-default grid carry a "_<grid>" suffix
# and the probe cache keys carry the grid string, so nothing collides.
GRID = os.environ.get("GRID", "16x8")
NU, ND = (int(v) for v in GRID.split("x"))
TAG = "" if GRID == "16x8" else f"_{GRID}"
G = NU * ND                    # cells
N_CLASSES = 3                  # empty / object 0 / object 1
N_OBJ = 2


def grid_edges(sim: dict) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges in (normalised u, 1/y) over the reachable region."""
    r = float(sim["radius"])
    u = np.linspace(-1.0, 1.0, NU + 1)
    d = np.linspace(1.0 / (float(sim["y_far"]) - r), 1.0 / (float(sim["y_near"]) + r), ND + 1)
    return u, d


def lateral_norm(pos: np.ndarray, sim: dict) -> np.ndarray:
    """u' = u / (1 − r/(scale·y)): the ray coordinate rescaled by a centre's reach at
    that depth, so u' = ±1 is a disc touching the frustum wall."""
    scale = float(sim["x_far"]) / float(sim["y_far"])
    r = float(sim["radius"])
    reach = 1.0 - r / (scale * np.maximum(pos[..., 1], 1e-6))
    return lateral(pos, sim) / np.maximum(reach, 1e-6)


def cell_of(pos: np.ndarray, sim: dict) -> np.ndarray:
    """(..., 2) world positions -> (...) int cell index in [0, G)."""
    ue, de = grid_edges(sim)
    u = lateral_norm(pos, sim)
    inv_y = 1.0 / np.maximum(pos[..., 1], 1e-6)
    iu = np.clip(np.searchsorted(ue, u, side="right") - 1, 0, NU - 1)
    idp = np.clip(np.searchsorted(de, inv_y, side="right") - 1, 0, ND - 1)
    return (iu * ND + idp).astype(np.int64)


def label_frames(pos: np.ndarray, sim: dict) -> tuple[np.ndarray, int]:
    """(..., N_OBJ, 2) positions -> (..., G) uint8 cell labels, nearer object winning a
    shared cell. Also returns the number of (frame, cell) conflicts resolved that way."""
    lead = pos.shape[:-2]
    cells = cell_of(pos, sim)                          # (..., N_OBJ)
    y = pos[..., 1]
    lab = np.zeros(lead + (G,), dtype=np.uint8)
    flat_lab = lab.reshape(-1, G)
    flat_c = cells.reshape(-1, N_OBJ)
    flat_y = y.reshape(-1, N_OBJ)
    order = np.argsort(-flat_y, axis=1)               # far first, so near overwrites
    rows = np.arange(flat_lab.shape[0])
    conflicts = int((flat_c[:, 0] == flat_c[:, 1]).sum())
    for k in range(N_OBJ):
        j = order[:, k]
        flat_lab[rows, flat_c[rows, j]] = (j + 1).astype(np.uint8)
    return lab, conflicts


def cell_centre_world(cell: int, sim: dict, y_ref: float | None = None):
    """Rough world coordinates of a cell centre (for labels / figures only)."""
    ue, de = grid_edges(sim)
    iu, idp = divmod(int(cell), ND)
    u = 0.5 * (ue[iu] + ue[iu + 1])
    inv_y = 0.5 * (de[idp] + de[idp + 1])
    y = 1.0 / inv_y
    scale = float(sim["x_far"]) / float(sim["y_far"])
    u = u * (1.0 - float(sim["radius"]) / (scale * y))           # undo the reach normalisation
    return float(u * scale * y), float(y)


def cell_of_frustum(u: np.ndarray, inv_y: np.ndarray, sim: dict) -> np.ndarray:
    """Cells from FRUSTUM-basis coordinates (u = x/(scale·y), 1/y) — e.g. a regression
    probe's read-out — so a continuous prediction can be scored on the grid axis."""
    ue, de = grid_edges(sim)
    scale = float(sim["x_far"]) / float(sim["y_far"])
    r = float(sim["radius"])
    reach = 1.0 - r * np.asarray(inv_y) / scale
    un = np.asarray(u) / np.maximum(reach, 1e-6)
    iu = np.clip(np.searchsorted(ue, un, side="right") - 1, 0, NU - 1)
    idp = np.clip(np.searchsorted(de, inv_y, side="right") - 1, 0, ND - 1)
    return (iu * ND + idp).astype(np.int64)


def labels_from_cells(cells: np.ndarray, y_world: np.ndarray) -> np.ndarray:
    """(..., N_OBJ) cells + (..., N_OBJ) depths -> (..., G) labels, nearer object winning."""
    lead = cells.shape[:-1]
    lab = np.zeros(lead + (G,), dtype=np.uint8)
    fl, fc, fy = lab.reshape(-1, G), cells.reshape(-1, N_OBJ), y_world.reshape(-1, N_OBJ)
    order = np.argsort(-fy, axis=1)
    rows = np.arange(fl.shape[0])
    for k in range(N_OBJ):
        j = order[:, k]
        fl[rows, fc[rows, j]] = (j + 1).astype(np.uint8)
    return lab
