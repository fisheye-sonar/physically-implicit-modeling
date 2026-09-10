"""The GRID probe target — discworld state as a categorical grid of cells (Othello's shape).

Canonicalised 2026-09-09 from ``experiments/grid_target_control/scripts/grid.py`` (2026-09-08),
verbatim in every formula; only the resolution moved from an environment variable into the
target NAME so the grid is one probe target among the others (``"grid-16x8"``), keyed and
tabled like a basis.

Othello's probe target is categorical (64 tiles × 3 classes); discworld's is continuous
(positions, regressed). This target re-expresses the discworld state as a grid of cells,
each labelled {0 empty, 1 centre of object 0 (bright), 2 centre of object 1 (dim)} — the
same shape of target as the Othello board probe — so the last nameable difference on the
STATE side of the comparison can be tested with new probes on the same trained model.

The grid lives in the frustum basis the canonical probes already use: lateral ray
coordinate u = x / (scale·y) in [-1, 1] and inverse depth 1/y in [1/y_far, 1/y_near],
binned UNIFORMLY there. That makes every cell the same size in the observation's own
coordinates (rays laterally, apparent width in depth — ``frustum.py`` explains why depth
enters the frame only as 1/y), and therefore larger at the back than the front in world
space. The observation-exact partition ("movement inside a cell changes no ray") would
need ~128 × 16 ≈ 2,000 cells; 16 × 8 = 128 is the coarsening that keeps the target at
Othello's scale (384 logits vs 192).

The grid covers the REACHABLE region: with ``always_in_frustum`` a disc centre keeps a
margin of one radius from every wall, so depth runs over [y_near + r, y_far − r] and the
lateral coordinate is normalised by its reach at that depth, u' = u / (1 − r/(scale·y)).
Without this the whole nearest depth row and the outer lateral columns at near depths
are cells nothing ever visits (29 of 128 on the eval split).

Two objects can share a cell only at the far plane (depth bins there span ~3 world
units); the NEARER object's label wins, since it is the one the observation shows.
``label_frames`` reports how often that happens.

⛔ An edit under this target is a categorical MOVE — (old cell → empty, new cell → the
object's class) — so a teleport that stays inside one cell is a no-op and is excluded
from the bench (``pim.environments.discworld.bench``), and ND IS applicable here (one
fixed change per case), unlike on the regression target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from pim.environments.discworld.frustum import lateral

N_CLASSES = 3                  # empty / object 0 / object 1
N_OBJ = 2
_NAME = re.compile(r"^grid-(\d+)x(\d+)$")


@dataclass(frozen=True)
class GridTarget:
    """One grid resolution: ``nu`` lateral × ``nd`` depth cells. Named ``grid-<nu>x<nd>``."""

    nu: int = 16
    nd: int = 8

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"grid-{self.nu}x{self.nd}"

    @property
    def g(self) -> int:
        """Number of cells."""
        return self.nu * self.nd

    @property
    def n_classes(self) -> int:
        return N_CLASSES

    @classmethod
    def parse(cls, target: str) -> "GridTarget | None":
        """``"grid-16x8"`` → ``GridTarget(16, 8)``; any other target name → None."""
        m = _NAME.match(str(target))
        return cls(int(m.group(1)), int(m.group(2))) if m else None

    # ── geometry ──────────────────────────────────────────────────────────────

    def edges(self, sim: dict) -> tuple[np.ndarray, np.ndarray]:
        """Bin edges in (normalised u, 1/y) over the reachable region."""
        r = float(sim["radius"])
        u = np.linspace(-1.0, 1.0, self.nu + 1)
        d = np.linspace(1.0 / (float(sim["y_far"]) - r), 1.0 / (float(sim["y_near"]) + r),
                        self.nd + 1)
        return u, d

    @staticmethod
    def lateral_norm(pos: np.ndarray, sim: dict) -> np.ndarray:
        """u' = u / (1 − r/(scale·y)): the ray coordinate rescaled by a centre's reach at
        that depth, so u' = ±1 is a disc touching the frustum wall."""
        scale = float(sim["x_far"]) / float(sim["y_far"])
        r = float(sim["radius"])
        reach = 1.0 - r / (scale * np.maximum(pos[..., 1], 1e-6))
        return lateral(pos, sim) / np.maximum(reach, 1e-6)

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        """(..., 2) world positions → (...) int cell index in [0, g)."""
        ue, de = self.edges(sim)
        u = self.lateral_norm(pos, sim)
        inv_y = 1.0 / np.maximum(pos[..., 1], 1e-6)
        iu = np.clip(np.searchsorted(ue, u, side="right") - 1, 0, self.nu - 1)
        idp = np.clip(np.searchsorted(de, inv_y, side="right") - 1, 0, self.nd - 1)
        return (iu * self.nd + idp).astype(np.int64)

    def cell_of_frustum(self, u: np.ndarray, inv_y: np.ndarray, sim: dict) -> np.ndarray:
        """Cells from FRUSTUM-basis coordinates (u = x/(scale·y), 1/y) — e.g. a regression
        probe's read-out — so a continuous prediction can be scored on the grid axis."""
        ue, de = self.edges(sim)
        scale = float(sim["x_far"]) / float(sim["y_far"])
        r = float(sim["radius"])
        reach = 1.0 - r * np.asarray(inv_y) / scale
        un = np.asarray(u) / np.maximum(reach, 1e-6)
        iu = np.clip(np.searchsorted(ue, un, side="right") - 1, 0, self.nu - 1)
        idp = np.clip(np.searchsorted(de, inv_y, side="right") - 1, 0, self.nd - 1)
        return (iu * self.nd + idp).astype(np.int64)

    def cell_centre_world(self, cell: int, sim: dict) -> tuple[float, float]:
        """Rough world coordinates of a cell centre (for labels / figures only)."""
        ue, de = self.edges(sim)
        iu, idp = divmod(int(cell), self.nd)
        u = 0.5 * (ue[iu] + ue[iu + 1])
        inv_y = 0.5 * (de[idp] + de[idp + 1])
        y = 1.0 / inv_y
        scale = float(sim["x_far"]) / float(sim["y_far"])
        u = u * (1.0 - float(sim["radius"]) / (scale * y))       # undo the reach normalisation
        return float(u * scale * y), float(y)

    # ── labels ────────────────────────────────────────────────────────────────

    def labels_from_cells(self, cells: np.ndarray, y_world: np.ndarray) -> np.ndarray:
        """(..., N_OBJ) cells + (..., N_OBJ) depths → (..., g) uint8 labels, the nearer
        object winning a shared cell."""
        lead = cells.shape[:-1]
        lab = np.zeros(lead + (self.g,), dtype=np.uint8)
        fl = lab.reshape(-1, self.g)
        fc, fy = cells.reshape(-1, N_OBJ), y_world.reshape(-1, N_OBJ)
        order = np.argsort(-fy, axis=1)                    # far first, so near overwrites
        rows = np.arange(fl.shape[0])
        for k in range(N_OBJ):
            j = order[:, k]
            fl[rows, fc[rows, j]] = (j + 1).astype(np.uint8)
        return lab

    def label_frames(self, pos: np.ndarray, sim: dict) -> tuple[np.ndarray, int]:
        """(..., N_OBJ, 2) positions → (..., g) uint8 cell labels, nearer object winning a
        shared cell. Also returns the number of (frame, cell) conflicts resolved that way."""
        cells = self.cell_of(pos, sim)                     # (..., N_OBJ)
        fc = cells.reshape(-1, N_OBJ)
        conflicts = int((fc[:, 0] == fc[:, 1]).sum())
        return self.labels_from_cells(cells, pos[..., 1]), conflicts

    def edit_cells(self, pos: np.ndarray, edit_object: np.ndarray, ef: int,
                   sim: dict) -> dict:
        """Per case, the categorical MOVE a teleport asks for: ``A`` the edited object's
        cell at frame ``ef − 1``, ``B`` its cell at ``ef``, ``cls`` its class (object + 1).
        ``pos`` (N, T, N_OBJ, 2) post-edit positions, ``edit_object`` (N,). Returns
        ``{"A", "B", "cls"}`` int64 arrays of shape (N,)."""
        idx = np.arange(len(edit_object))
        j = np.asarray(edit_object, dtype=int)
        return {"A": self.cell_of(pos[idx, ef - 1, j], sim),
                "B": self.cell_of(pos[idx, ef, j], sim),
                "cls": (j + 1).astype(np.int64)}


CANONICAL = GridTarget(16, 8)
