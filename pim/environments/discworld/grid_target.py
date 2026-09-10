"""CATEGORICAL probe targets for discworld — the state as cells (Othello's shape of target).

Othello's probe target is categorical (64 tiles × 3 classes); discworld's canonical one is
continuous (positions, regressed). A categorical target re-expresses the discworld state as
a set of cells, each labelled {0 empty, 1 centre of object 0 (bright), 2 centre of object 1
(dim)} — the same shape as the Othello board probe — so the STATE side of the comparison
can be tested with new probes on the same trained model. Two families, one interface:

``GridTarget`` — ``"grid-<nu>x<nd>"`` (2026-09-08, canonicalised 2026-09-09 from
``experiments/grid_target_control/scripts/grid.py``, formulas verbatim). A product grid in
the frustum basis the canonical probes use: lateral ray coordinate u = x / (scale·y) in
[-1, 1] and inverse depth 1/y, binned UNIFORMLY, so every cell is the same size in the
observation's own coordinates (rays laterally, apparent width in depth — ``frustum.py``
explains why depth enters the frame only as 1/y) and larger at the back in world space.
16 × 8 = 128 cells kept the 128-ray noiseless instance at Othello's scale (384 logits); the
observation-exact partition there would need 2,883 cells.

``AppearanceTarget`` — ``"appearance"`` (2026-09-09). The observation-exact partition
itself: two positions are in the same cell iff a single disc there LIGHTS THE SAME RAYS.
With binary occupancy and a fixed reflectivity per object, a disc's appearance is a
contiguous run of rays (first, last); lateral position sets the run's centre, depth its
length. On dw-8ray (radius 1.0, 8 kept rays) exactly 30 runs are realisable over the
reachable region (lengths 1–5; the empty appearance never occurs), i.e. 90 logits — the
finest target a single frame can resolve, and the coarsest at which an edit is visible in
the output. Cells are ray-based by construction and are NOT a product grid: a two-ray run
spans depths ~6–11. Variants, for the resolution sweep:
    ``"appearance-d<k>"``  each run cell split into k depth bands, uniform in 1/y over the
                           depths that run occupies (does resolving MORE than the frame does
                           help the editors? the model may carry depth from motion)
    ``"appearance-lat"``   runs merged by CENTRE only (start + end), depth dropped — coarser
                           than the observation resolves
The realisable runs and their depth ranges come from a deterministic dense sweep of the
reachable region with the analytic ray test (``covered_rays``, the renderer's own
ray–disc intersection, gated equal to ``render_frame`` in tests).

Both cover the REACHABLE region: with ``always_in_frustum`` a disc centre keeps a margin of
one radius from every wall. Two objects can share a cell (the grid: only at the far plane,
0.05% of frames; appearance on dw-8ray: 0.31%); the NEARER object's label wins, since it
is the one the observation shows. ``label_frames`` reports how often that happens.

⛔ An edit under any of these is a categorical MOVE — (old cell → empty, new cell → the
object's class) — so a teleport that stays inside one cell is a no-op and is excluded from
the bench (``pim.environments.discworld.bench.grid_selection``), and ND IS applicable (one
fixed change per case), unlike on the regression target.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import numpy as np

from pim.environments.discworld.frustum import lateral

N_CLASSES = 3                  # empty / object 0 / object 1
N_OBJ = 2
_GRID = re.compile(r"^grid-(\d+)x(\d+)$")
_APP = re.compile(r"^appearance(?:-d(\d+)|-(lat))?$")


class CategoricalTarget:
    """The interface every categorical target shares (``cell_of`` is the family-specific part)."""

    n_classes = N_CLASSES

    @property
    def name(self) -> str:                       # pragma: no cover — abstract
        raise NotImplementedError

    def n_cells(self, sim: dict) -> int:         # pragma: no cover — abstract
        raise NotImplementedError

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:   # pragma: no cover
        """(..., 2) world positions → (...) int cell index in [0, n_cells)."""
        raise NotImplementedError

    def labels_from_cells(self, cells: np.ndarray, y_world: np.ndarray, sim: dict) -> np.ndarray:
        """(..., N_OBJ) cells + (..., N_OBJ) depths → (..., G) uint8 labels, the nearer
        object winning a shared cell."""
        g = self.n_cells(sim)
        lead = cells.shape[:-1]
        lab = np.zeros(lead + (g,), dtype=np.uint8)
        fl = lab.reshape(-1, g)
        fc, fy = cells.reshape(-1, N_OBJ), y_world.reshape(-1, N_OBJ)
        order = np.argsort(-fy, axis=1)                    # far first, so near overwrites
        rows = np.arange(fl.shape[0])
        for k in range(N_OBJ):
            j = order[:, k]
            fl[rows, fc[rows, j]] = (j + 1).astype(np.uint8)
        return lab

    def label_frames(self, pos: np.ndarray, sim: dict) -> tuple[np.ndarray, int]:
        """(..., N_OBJ, 2) positions → (..., G) uint8 cell labels, nearer object winning a
        shared cell. Also returns the number of (frame, cell) conflicts resolved that way."""
        cells = self.cell_of(pos, sim)                     # (..., N_OBJ)
        fc = cells.reshape(-1, N_OBJ)
        conflicts = int((fc[:, 0] == fc[:, 1]).sum())
        return self.labels_from_cells(cells, pos[..., 1], sim), conflicts

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


# ── the product grid ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GridTarget(CategoricalTarget):
    """One grid resolution: ``nu`` lateral × ``nd`` depth cells. Named ``grid-<nu>x<nd>``."""

    nu: int = 16
    nd: int = 8

    @property
    def name(self) -> str:
        return f"grid-{self.nu}x{self.nd}"

    @property
    def g(self) -> int:
        """Number of cells (independent of the instance)."""
        return self.nu * self.nd

    def n_cells(self, sim: dict) -> int:
        return self.g

    @classmethod
    def parse(cls, target: str) -> "GridTarget | None":
        """``"grid-16x8"`` → ``GridTarget(16, 8)``; any other target name → None."""
        m = _GRID.match(str(target))
        return cls(int(m.group(1)), int(m.group(2))) if m else None

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


CANONICAL = GridTarget(16, 8)


# ── the appearance partition ──────────────────────────────────────────────────


def covered_rays(pos: np.ndarray, sim: dict) -> np.ndarray:
    """(..., 2) single-disc centres → (..., R_kept) bool: which KEPT rays that disc lights.

    The renderer's own ray–disc test (``renderer.render_frame``): rays fan out with
    directions ``(s·scale, 1)`` normalised, s uniform in [-1, 1] over ``obs_res`` rays; a
    ray hits iff its discriminant is non-negative and the front intersection lies within
    [y_near, y_far] (the near-plane clamp case never arises for an always-in-frustum disc,
    whose centre keeps a margin of one radius). The two wall rays are dropped when the
    instance asks for it (``drop_edge_rays``), exactly as the renderer does.
    """
    p = np.asarray(pos)
    R = int(sim["obs_res"])
    scale = float(sim["x_far"]) / float(sim["y_far"])
    s = np.linspace(-1.0, 1.0, R)
    dx, dy = s * scale, np.ones(R)
    nrm = np.hypot(dx, dy)
    dx, dy = dx / nrm, dy / nrm
    cx, cy = p[..., 0:1], p[..., 1:2]                       # (..., 1), the caller's dtype
    b = dx * cx + dy * cy                                   # (..., R) float64
    # |c|² − r² in the POSITIONS' dtype, as the renderer computes it (float32 positions and
    # radii give a float32 C) — a grazing ray can flip otherwise (1 in 12,000 on dw-noiseless)
    r = np.asarray(sim["radius"], dtype=p.dtype if np.issubdtype(p.dtype, np.floating) else np.float64)
    C = (cx ** 2 + cy ** 2 - r ** 2).astype(np.float64)
    disc = b ** 2 - C
    t_front = b - np.sqrt(np.maximum(disc, 0.0))
    y_front = dy * t_front
    hit = (disc >= 0) & (t_front > 1e-9) & (y_front >= float(sim["y_near"])) & (y_front <= float(sim["y_far"]))
    return hit[..., 1:-1] if sim.get("drop_edge_rays", False) else hit


def _sim_key(sim: dict) -> tuple:
    return tuple(float(sim[k]) for k in ("radius", "y_near", "y_far", "x_far")) + \
        (int(sim["obs_res"]), bool(sim.get("drop_edge_rays", False)))


@functools.lru_cache(maxsize=None)
def _runs_of(sim_key: tuple, n_side: int = 500):
    """Deterministic dense sweep of the reachable region → the realisable (first, last)
    runs in a fixed order, each with the depth range it occupies. Cached per geometry."""
    r, y_near, y_far, x_far, R, drop = sim_key
    sim = {"radius": r, "y_near": y_near, "y_far": y_far, "x_far": x_far, "obs_res": R,
           "drop_edge_rays": drop}
    scale = x_far / y_far
    ys = np.linspace(y_near + r, y_far - r, n_side)
    xs = np.linspace(-1.0, 1.0, n_side)
    Y = np.repeat(ys[:, None], n_side, 1)
    X = xs[None, :] * (scale * Y - r)                       # inside the reach at each depth
    hit = covered_rays(np.stack([X, Y], -1), sim)          # (n, n, Rk)
    first = hit.argmax(-1)
    last = hit.shape[-1] - 1 - hit[..., ::-1].argmax(-1)
    any_ = hit.any(-1)
    assert any_.all(), "a reachable position lights no ray — the appearance partition needs r large enough"
    assert (hit.sum(-1) == last - first + 1).all(), "non-contiguous run"
    code = first * hit.shape[-1] + last
    runs, depth = [], {}
    for c in np.unique(code):
        m = code == c
        runs.append((int(c // hit.shape[-1]), int(c % hit.shape[-1])))
        depth[runs[-1]] = (float(Y[m].min()), float(Y[m].max()))
    runs.sort(key=lambda fl: (fl[1] - fl[0], fl[0]))        # short runs (far) first, then left→right
    return tuple(runs), depth, hit.shape[-1]


def _nearest_run(runs: tuple, f: int, la: int) -> int:
    """Index of the realisable run nearest to (f, l): same centre first, then smallest
    |Δf| + |Δl|, ties to the earlier run in cell order."""
    best, key = 0, None
    for i, (rf, rl) in enumerate(runs):
        k = (abs((rf + rl) - (f + la)), abs(rf - f) + abs(rl - la))
        if key is None or k < key:
            best, key = i, k
    return best


@dataclass(frozen=True)
class AppearanceTarget(CategoricalTarget):
    """The observation-exact partition: cell = the run of rays a disc lights, optionally
    split into ``depth_bands`` depth bands (uniform in 1/y within the run's depth range) or
    merged by run centre (``lateral_only``)."""

    depth_bands: int = 1
    lateral_only: bool = False

    @property
    def name(self) -> str:
        if self.lateral_only:
            return "appearance-lat"
        return "appearance" if self.depth_bands == 1 else f"appearance-d{self.depth_bands}"

    @classmethod
    def parse(cls, target: str) -> "AppearanceTarget | None":
        m = _APP.match(str(target))
        if not m:
            return None
        if m.group(2):
            return cls(lateral_only=True)
        return cls(depth_bands=int(m.group(1)) if m.group(1) else 1)

    def runs(self, sim: dict) -> tuple:
        """The realisable (first, last) runs on this instance, in cell order."""
        return _runs_of(_sim_key(sim))[0]

    def _centres(self, sim: dict) -> list[int]:
        return sorted({f + la for f, la in self.runs(sim)})

    def n_cells(self, sim: dict) -> int:
        if self.lateral_only:
            return len(self._centres(sim))
        return len(self.runs(sim)) * self.depth_bands

    # positions per chunk of the ray–disc test: (..., R) float64 intermediates at 128 rays
    # are ~1 KB per position, so 2^18 positions ≈ 0.3 GB each; labelling the 15.6 M
    # positions of the probe corpus in one shot was ~60 GB and OOM-killed a unit (2026-09-10)
    CHUNK = 1 << 18

    def cell_of(self, pos: np.ndarray, sim: dict) -> np.ndarray:
        p = np.asarray(pos, np.float64)
        flat = p.reshape(-1, 2)
        if flat.shape[0] <= self.CHUNK:
            return self._cell_of_flat(flat, sim).reshape(p.shape[:-1])
        out = np.concatenate([self._cell_of_flat(flat[i: i + self.CHUNK], sim)
                              for i in range(0, flat.shape[0], self.CHUNK)])
        return out.reshape(p.shape[:-1])

    def _cell_of_flat(self, p: np.ndarray, sim: dict) -> np.ndarray:
        runs, depth, rk = _runs_of(_sim_key(sim))
        hit = covered_rays(p, sim)
        first = hit.argmax(-1)
        last = rk - 1 - hit[..., ::-1].argmax(-1)
        code = first * rk + last
        table = np.full(rk * rk, -1, np.int64)
        for i, (f, la) in enumerate(runs):
            table[f * rk + la] = i
        run_idx = table[code]
        if (run_idx < 0).any():
            # A run the sweep did not see: a grazing ray flipped by float32 rounding, so the
            # disc lights one ray more or fewer than any swept position (4 in 2 M positions
            # on dw-noiseless; never on dw-8ray). Snap to the nearest realisable run.
            miss = run_idx < 0
            if not hit[miss].any(-1).all():
                raise ValueError("a position lights no ray — outside the reachable region")
            snapped = np.array([_nearest_run(runs, int(c // rk), int(c % rk))
                                for c in np.unique(code[miss])])
            lut = dict(zip(np.unique(code[miss]).tolist(), snapped.tolist()))
            run_idx = run_idx.copy()
            run_idx[miss] = [lut[int(c)] for c in code[miss]]
            first = np.where(miss, np.array([runs[i][0] for i in run_idx]), first)
            last = np.where(miss, np.array([runs[i][1] for i in run_idx]), last)
        if self.lateral_only:
            cen = {c: i for i, c in enumerate(self._centres(sim))}
            ctab = np.full(2 * rk, -1, np.int64)
            for c, i in cen.items():
                ctab[c] = i
            return ctab[first + last]
        if self.depth_bands == 1:
            return run_idx
        # band by 1/y, uniform within the run's own depth range
        lo = np.array([1.0 / depth[rn][1] for rn in runs])   # 1/ymax
        hi = np.array([1.0 / depth[rn][0] for rn in runs])   # 1/ymin
        inv_y = 1.0 / np.maximum(p[..., 1], 1e-6)
        frac = (inv_y - lo[run_idx]) / np.maximum(hi[run_idx] - lo[run_idx], 1e-9)
        band = np.clip((frac * self.depth_bands).astype(np.int64), 0, self.depth_bands - 1)
        return run_idx * self.depth_bands + band


def categorical_target(target: str) -> "CategoricalTarget | None":
    """The categorical target a name denotes (``grid-…``, ``appearance…``), or None for a
    regression target (``pos`` / ``full``). Every branch in the pipeline goes through this."""
    return GridTarget.parse(target) or AppearanceTarget.parse(target)
