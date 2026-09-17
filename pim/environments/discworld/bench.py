"""The discworld editability bench: the canonical edit set, its targets and ray zones.

The counterpart of ``pim.environments.othello.bench`` (split out of ``arms.py`` 2026-09-07
so the two environments read the same way): this module is the EDIT SET — the same 192
mid-sequence teleports from the instance's ``eval/edits.h5``, warmed into a model state,
with probe targets in the requested basis and ``pim.metrics.zone_editability``'s ray zones.
Probes, rollouts and editor arms live in ``arms.py``. That constancy is what makes every
Edit Index in the project comparable, across models, bases, and probe corpora.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.environments import layout
from pim.environments.discworld.grid_target import (
    CategoricalTarget, FactorisedTarget, categorical_target, selection_target, snapped_target)
from pim.metrics.zone_editability import build_edit_zones

N_OBJ, EF, K_ROLL, SEED = 2, 20, 15, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_REPO = layout.REPO
# The canonical edit set's directory on the default instance — kept as the module default
# for callers that name neither an instance nor a directory. Resolved through
# ``pim.environments.layout`` (v2: ``edits/v1/``; v1: ``eval/``).
DATA = layout.edits_dir("discworld", layout.DEFAULT_INSTANCE["discworld"])


def _edit_set(data_dir: Path | None, instance: str | None) -> tuple[Path, Path, str | None]:
    """(edits.h5, selection.json, instance) for a bench request. ``instance`` names an
    instance directly (the v2 form). ``data_dir`` is the older form: an instance's edit
    directory (``<inst>/eval`` under v1, ``<inst>/edits/v1`` under v2 — both map onto the
    instance, so old call sites keep working after the move) or ANY directory holding an
    ``edits.h5`` (a pilot under experiments/), which is used as is."""
    if instance is not None:
        inst = instance
    elif data_dir is None:
        inst = layout.DEFAULT_INSTANCE["discworld"]
    else:
        li = layout.legacy_edits_instance(data_dir)
        if li is None:
            dd = Path(data_dir)
            return dd / "edits.h5", dd.parent / "edits_selection.json", None
        inst = li[1]
    return (layout.edits_file("discworld", inst), layout.edits_selection("discworld", inst), inst)


@dataclass
class Bench:
    """The edits split, warmed and scored-against: every editability number's ground."""

    obs: np.ndarray  # (N, T, R) noisy observations
    gt_roll: np.ndarray  # (N, K, R) clean post-edit ground truth from the edit frame
    zones: object  # target / ghost / differing ray masks
    tgt: torch.Tensor  # (N, d_out) the probe target the edit asks for
    change_mask: torch.Tensor  # (N, d_out) bool — the EDITED object's dims only
    out_dims: list[int]  # the read-out rows the edit is allowed to move
    state: object  # the model state warmed on obs[:, :EF]
    n: int
    # the world the oracle editors need (added 2026-09-07; default None keeps older
    # positional constructions valid)
    pos: np.ndarray | None = None          # (N, T, N_OBJ, 2) post-edit positions
    vel: np.ndarray | None = None          # (N, T, N_OBJ, 2)
    edit_object: np.ndarray | None = None  # (N,)
    sim: dict | None = None                # the instance's sim config
    # the probe target's KIND (2026-09-09): "regression" — tgt holds values in the probe's
    # output units; "classification" — tgt holds (N, d_out) long labels and `cells` the
    # categorical MOVE per case ({"A", "B", "cls"} long tensors: old cell, new cell, class)
    kind: str = "regression"
    cells: dict | None = None
    # a FACTORISED categorical target's move (2026-09-10): per case the edited object's tiles
    # and their classes before / after — {"tile", "old", "new"} (N, F) long; `cells` is None
    moves: dict | None = None
    selection: dict | None = None          # which cases were scored, when not the first n


def _to_basis(pos, vel, sim, basis_name):
    """World (x,y[,vx,vy]) -> the named basis. None/'cartesian' is a no-op."""
    if basis_name in (None, "cartesian"):
        return pos, vel
    from pim.environments.discworld.frustum import basis as fb

    return fb(pos, vel, sim, depth=basis_name)


def selection_path(data_dir: Path | None = None, instance: str | None = None) -> Path:
    """An instance's FILTERED edit-case list, if it has one (``edits/v1/selection.json``;
    ``edits_selection.json`` beside ``eval/`` under layout v1).

    Written by ``experiments/interface_ablation/edits_audit/scripts/make_selection.py``.
    It exists for dw-8ray only (2026-09-08): with 8 rays a teleport renders an IDENTICAL
    frame in 20% of cases and moves a single ray in another 22%, so a first-n bench scores
    ~160 of 192 cases and many of the rest are one-ray marginals. The 128-ray instances do
    not need it (0.5% degenerate, mean 27 rays differing). ⛔ dw-blink's 19% zero-differing
    cases are the MID-BLACKOUT population — real cases scored at reappearance by
    experiments/blink_ablation — and must never be filtered away here.
    The SAME file serves the ray-zone and the frames-as-tokens bench, so the interface
    ablation stays paired case for case.
    """
    return _edit_set(data_dir, instance)[1]


def grid_selection(data_dir: Path | None, n: int, grid: CategoricalTarget,
                   instance: str | None = None) -> tuple[np.ndarray, dict]:
    """The GRID target's bench: the first ``n`` cases whose teleport CHANGES CELL.

    Under a categorical target a teleport that stays inside one cell asks for no change at
    all, so it cannot be scored as an edit; such cases are skipped and the next ones taken
    (2.1% on dw-noiseless at 16 × 8). Reads positions only — no frames — so it is cheap.
    Returns the case indices and a record for ``scores.json`` (rule, counts)."""
    with h5py.File(_edit_set(data_dir, instance)[0], "r") as f:
        pos = f["positions"][:, EF - 1: EF + 1, :N_OBJ, :].astype(np.float32)   # (M, 2, N_OBJ, 2)
        vel = f["velocities"][:, EF - 1, :N_OBJ, :].astype(np.float32)          # (M, N_OBJ, 2)
        eobj = f["edit_object"][:].astype(int)
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    # frame 1 of the slice becomes the PRE-dynamics target state (see bench_arrays): the
    # edited object at pos[EF] − v·dt, the other object at its current position
    ar = np.arange(len(pos))
    pre_dyn = pos[:, 0].copy()
    pre_dyn[ar, eobj] = pos[ar, 1, eobj] - vel[ar, eobj] * float(sim["dt"])
    pos[:, 1] = pre_dyn
    mv = grid.edit_cells(pos, eobj, ef=1, sim=sim)          # frame 1 of the 2-frame slice = the target
    valid = np.where(mv["A"] != mv["B"])[0]
    sel = valid[:n]
    scanned = int(sel[-1]) + 1 if len(sel) else 0
    return sel, {"rule": f"first {n} cases whose teleport changes a {grid.name} cell "
                         f"(pre-dynamics target state, 2026-09-12)",
                 "n": int(len(sel)), "scanned": scanned,
                 "dropped_same_cell": int(scanned - len(sel))}


def bench_from_arrays(obs: np.ndarray, pos: np.ndarray, vel: np.ndarray, eobj: np.ndarray,
                      clean: np.ndarray, sim: dict, blink: np.ndarray | None, *, target: str,
                      basis_name: str, selection: dict | None = None, edit_frame: int = EF) -> dict:
    """The bench dict from ARRAYS — the target / change-mask / zone construction of
    ``bench_arrays``, factored out (2026-09-17) so a synthetic, matched scenario (the paper's
    qualitative figure: one trajectory rendered under every instance's geometry) goes
    through exactly the construction the scorer uses. ``obs`` (n, T, R) the frames the model
    is fed, ``clean`` (n, T, R) the clean render, ``pos`` / ``vel`` (n, T, N_OBJ, 2) POST-edit,
    ``eobj`` (n,), ``sim`` the instance's sim dict, ``blink`` (n, T, N_OBJ) bool or None."""
    grid, snap = categorical_target(target), snapped_target(target)
    n = obs.shape[0]
    gt_roll = clean[:, EF: EF + K_ROLL, :]
    zones = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF],
                             pre_vel=vel[:, EF - 1], edit_object=eobj, sim=sim,
                             n_obj=N_OBJ, traj_pos=pos[:, EF: EF + K_ROLL],
                             gt_edited_traj=gt_roll,
                             blink_visible=None if blink is None
                             else blink[:, EF - 1: EF + K_ROLL + 1])
    # ⛔ The ZONES stay in world space — they are ray masks over the observation and do
    # not depend on how the state is coordinatised. Only the PROBE TARGET changes basis,
    # so the Edit Index remains directly comparable across bases.
    #
    # ⛔ THE WRITE TARGET IS THE PRE-DYNAMICS STATE (2026-09-12, Sevan). The probe at the
    # edit point reads the state that rendered the LAST CONSUMED frame (EF−1); the model's
    # next output is ITS OWN dynamics step ahead of that state. The edit asks the object to
    # APPEAR at pos[EF] in the next frame, so the state to write is the one that produces
    # it: pos[EF] − v·dt for the edited object, and the CURRENT state (frame EF−1) for
    # everything else — a hold, not a one-step advance. The references (gt_edited,
    # gt_unedited, gt_roll) are unchanged: they are the post-dynamics renders the model's
    # next output is compared with. Until this date the target was the post-dynamics state
    # itself, one velocity step ahead of anything a write could produce: a PERFECT write
    # scored +0.70 on dw-noiseless (128 rays; 0.15 radii per step) and +0.92 on dw-8ray
    # against a +1 ceiling (GOTCHAS 2026-09-12). Othello never had the problem — its
    # probe reads the board after the last move and uniform-over-legal IS the next step.
    dt = float(sim["dt"])
    ar1 = np.arange(n)
    pre_dyn = pos[:, EF - 1].copy()                                     # (n, N_OBJ, 2)
    pre_dyn[ar1, eobj] = pos[ar1, EF, eobj] - vel[ar1, EF - 1, eobj] * dt
    pos_t = pos.copy()
    pos_t[:, EF] = pre_dyn            # frame EF read as the PRE-dynamics target state below
    cells = moves = None
    if isinstance(grid, FactorisedTarget):
        # Factorised MOVE: the edited object's factor tiles take the classes of the
        # pre-dynamics target state; only the tiles whose class changes are asked to move.
        cur, _ = grid.label_frames(pos[:, EF - 1], sim)                  # (n, tiles) int64
        moves = grid.edit_moves(pos_t, eobj, EF, sim)
        ar = np.arange(n)[:, None]
        y = cur.copy()
        y[ar, moves["tile"]] = moves["new"]
        cm = np.zeros((n, grid.n_tiles_on(sim)), bool)
        cm[ar, moves["tile"]] = moves["new"] != moves["old"]
        out_dims = []
    elif grid is not None:
        # Categorical MOVE: the labels the model should read after the edit are the
        # current frame's labels with the object gone from its old cell A and present in
        # its new cell B; only those two cells are asked to change (the rest hold).
        cur, _ = grid.label_frames(pos[:, EF - 1], sim)                  # (n, G) uint8
        cells = grid.edit_cells(pos_t, eobj, EF, sim)                    # B = the pre-dynamics cell
        ar = np.arange(n)
        y = cur.astype(np.int64)
        y[ar, cells["A"]] = 0
        y[ar, cells["B"]] = cells["cls"]
        cm = np.zeros((n, grid.n_cells(sim)), bool)
        cm[ar, cells["A"]] = True
        cm[ar, cells["B"]] = True
        out_dims = []            # per-case rows, not a shared set — see arms.nanda_rollout
    else:
        bp, bv = _to_basis(pre_dyn, vel[:, EF - 1], sim, basis_name)
        base = target
        if snap is not None:
            # Snapped regression (2026-09-10): the edit asks for the CENTRE of the new cell,
            # in frustum coordinates; velocities (``full@…``) stay the basis velocities.
            if basis_name != "frustum":
                raise ValueError(f"{target} is defined in the frustum basis, got {basis_name!r}")
            bp, base = snap.snap(pre_dyn, sim).astype(np.float32), snap.base
        y = bp.reshape(n, -1)
        if base == "full":
            y = np.concatenate([y, bv.reshape(n, -1)], axis=1)
        # The edit moves ONE object; everything else is a hold-the-rest constraint. Marking
        # too many dims would quietly turn a targeted edit into a whole-state overwrite.
        d_out = y.shape[1]
        cm = np.zeros((n, d_out), bool)
        cm[np.arange(n), 2 * eobj] = True
        cm[np.arange(n), 2 * eobj + 1] = True
        if base == "full":
            cm[np.arange(n), 2 * N_OBJ + 2 * eobj] = True
            cm[np.arange(n), 2 * N_OBJ + 2 * eobj + 1] = True
        out_dims = sorted({int(i) for i in np.where(cm.any(0))[0]})
    assert int(edit_frame) == EF, f"edit_frame {edit_frame}: every thread number assumes {EF}"
    return dict(obs=obs, pos=pos, vel=vel, edit_object=eobj, clean=clean, sim=sim,
                gt_roll=gt_roll, zones=zones, y=y, change_mask=cm, out_dims=out_dims, n=n,
                blink_visible=blink, kind="classification" if grid else "regression",
                cells=cells, moves=moves, selection=selection)


def bench_arrays(n: int = 192, target: str = "pos", basis_name: str = "cartesian",
                 data_dir: Path | None = None, select: np.ndarray | None = None,
                 use_selection: bool = True, instance: str | None = None) -> dict:
    """The edit set's arrays and zones, model-free (factored out of ``load_bench``,
    2026-09-05, so a token model — ``token_bench`` — scores the SAME cases, targets
    and zones without a frame-space state).

    ``instance`` names the instance whose canonical bench (``edits/v1``) to load — the
    layout-v2 form (2026-09-10). ``data_dir`` is the older form and still works: an
    instance's edit directory maps onto the instance; any other directory holding an
    ``edits.h5`` (a pilot) is read as is. See ``_edit_set``.

    ``select`` (case indices into the edits split) replaces "the first ``n`` cases" —
    for subset benches (e.g. the dw-blink reappearance cases, 2026-09-07). The
    canonical bench is always ``select=None``.

    ``target`` is ``"pos"`` / ``"full"`` (regression, in ``basis_name``), a SNAPPED
    regression target ``"pos@<partition>"`` (2026-09-10: positions replaced by their cell
    centre in the frustum basis; the case list is ``grid_selection`` on that partition, the
    branch otherwise the regression one) or a grid name
    such as ``"grid-16x8"`` (classification, 2026-09-09): then ``y`` holds the (N, cells)
    labels at the edit frame, ``change_mask`` marks the old and new cell, ``cells`` the
    per-case move, and the case list is ``grid_selection`` (teleports that change cell).
    The zones are unchanged either way — they never depend on the probe target.

    Uses ``pim.environments.discworld.loading``: ``clean_obs`` is RECONSTRUCTED from
    stored ids/reflectivities, not stored — reading the h5 directly gets a KeyError.
    On a blink instance the split's ``blink_visible`` schedule is handed to the zone
    construction so the reference worlds carry the same blackouts and markers.
    """
    from pim.environments.discworld.loading import load_edits

    edits_h5, _sp, _inst = _edit_set(data_dir, instance)
    selection = None
    sel_target = selection_target(target)      # the grid, or a snapped target's partition
    if sel_target is not None and select is None:
        select, selection = grid_selection(data_dir, n, sel_target, instance=instance)
    if select is None and use_selection:                 # the instance's filtered case list
        if _sp.exists():
            _sel = json.loads(_sp.read_text())
            select = np.asarray(_sel["select"], dtype=int)[:n]
            # RECORD that the filtered list was used (2026-09-14): the regression blocks of
            # every discworld run had `bench_selection: None` while the selection WAS applied,
            # so the record could not say which cases a number came from.
            selection = {"file": str(_sp.relative_to(_REPO)) if _sp.is_relative_to(_REPO) else str(_sp),
                         "rule": _sel.get("rule"), "n": int(len(select)),
                         **{k: _sel[k] for k in ("min_rays", "pool", "stats") if k in _sel}}
    # the edits split alone — its own config_json carries the sim config, so the 188 MB
    # test split is never decompressed just to read a dict (2026-09-07)
    b = load_edits(edits_h5, n_obj_keep=N_OBJ)
    sl = slice(None, n) if select is None else np.asarray(select, dtype=int)
    obs = b.obs[sl].astype(np.float32)
    pos = b.positions[sl, :, :N_OBJ, :].astype(np.float32)
    eobj = b.edit_object[sl].astype(int)
    clean = b.clean_obs[sl].astype(np.float32)
    n = obs.shape[0]
    with h5py.File(b.h5_path, "r") as f:
        vel = f["velocities"][:, :, :N_OBJ, :].astype(np.float32)[sl]
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    blink = None if b.blink_visible is None else b.blink_visible[sl]
    return bench_from_arrays(obs, pos, vel, eobj, clean, sim, blink, target=target,
                             basis_name=basis_name, selection=selection,
                             edit_frame=int(getattr(b, "edit_frame", EF)))


def load_bench(model, n: int = 192, target: str = "pos",
               basis_name: str = "cartesian", data_dir: Path | None = None,
               select: np.ndarray | None = None, use_selection: bool = True,
               instance: str | None = None) -> Bench:
    """Warm ``model`` on the edits split and build the ground-truth zones (``bench_arrays``)."""
    a = bench_arrays(n, target, basis_name, data_dir, select=select, use_selection=use_selection,
                     instance=instance)
    return bench_of(model, a)


def bench_of(model, a: dict) -> Bench:
    """Warm ``model`` on a bench dict's frames (``bench_arrays`` / ``bench_from_arrays``)."""
    state = model.state_from_obs(torch.from_numpy(a["obs"][:, :EF]).float().to(DEV))
    tgt = torch.from_numpy(a["y"]).to(DEV)
    tgt = tgt.long() if a["kind"] == "classification" else tgt.float()
    _long = lambda d: (None if d is None                                   # noqa: E731
                       else {k: torch.from_numpy(v).long().to(DEV) for k, v in d.items()})
    return Bench(a["obs"], a["gt_roll"], a["zones"], tgt,
                 torch.from_numpy(a["change_mask"]).to(DEV), a["out_dims"], state, a["n"],
                 pos=a["pos"], vel=a["vel"], edit_object=a["edit_object"], sim=a["sim"],
                 kind=a["kind"], cells=_long(a["cells"]), moves=_long(a["moves"]),
                 selection=a["selection"])


# ── which read-outs an edit drives ───────────────────────────────────────────
#
# Discworld fits ONE probe set per basis, on the FULL state, and the editability sweep
# asks that probe for position alone AND for everything; the better of the two is the
# reported number (2026-09-01). Nothing was lost in retiring the pos-only probes: for the
# LINEAR probe the position rows of a full-state least-squares fit are BIT-IDENTICAL to a
# position-only fit — multi-output lstsq decomposes per output dimension, so fitting
# velocity alongside cannot perturb the position rows (verified on cached probes,
# max|W_full[:4] − W_pos| = 0.0 in both bases). "pos" therefore REPRODUCES the retired
# pos-only probe exactly for PI and ND. The MLP does not decompose — its hidden layer
# couples the outputs — so for GS the two are genuinely different probes, which is the
# reason to sweep both rather than assume.

DIM_SETS: dict[str, tuple[int, ...] | None] = {
    "pos": tuple(range(2 * N_OBJ)),   # every object's position
    "all": None,                      # every read-out the probe has
}


def dim_idx(dims: str):
    """Name -> read-out indices (None = all of them)."""
    if dims not in DIM_SETS:
        raise KeyError(f"dims must be one of {sorted(DIM_SETS)}, got {dims!r}")
    return DIM_SETS[dims]


def restrict_mask(cm: torch.Tensor, dims: str) -> torch.Tensor:
    """A change mask keeping only ``dims``. The dropped read-outs do NOT leave the loss:
    ``build_edit_spec`` holds every unmasked dim at its pre-edit value, so they become
    hold-the-rest constraints — which is what "edit position only" should mean."""
    idx = dim_idx(dims)
    if idx is None:
        return cm
    keep = torch.zeros(cm.shape[1], dtype=torch.bool, device=cm.device)
    keep[list(idx)] = True
    return cm & keep


def full_state_pair(pos: np.ndarray, vel: np.ndarray, edit_object: np.ndarray, sim: dict,
                    basis_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The FULL state (position + velocity of every object, in ``basis_name``) BEFORE and
    AFTER the edit, per case — the inverse-map editor's input pair (2026-09-15).

    ``s_pre`` is the state that rendered the last consumed frame (EF−1). ``s_post`` is the
    PRE-DYNAMICS target state (the 2026-09-12 write target, see ``bench_arrays``): the
    edited object at pos[EF] − v·dt, everything else held — so on the regression ``full``
    bench ``s_post`` equals ``Bench.tgt`` exactly, and on a categorical bench it is the
    same state the categorical move was derived from. Returns two (N, 4·N_OBJ) arrays.
    """
    n = len(pos)
    ar = np.arange(n)
    eobj = np.asarray(edit_object, int)
    pre_dyn = pos[:, EF - 1].copy()
    pre_dyn[ar, eobj] = pos[ar, EF, eobj] - vel[ar, EF - 1, eobj] * float(sim["dt"])
    bp0, bv0 = _to_basis(pos[:, EF - 1], vel[:, EF - 1], sim, basis_name)
    bp1, bv1 = _to_basis(pre_dyn, vel[:, EF - 1], sim, basis_name)
    s_pre = np.concatenate([bp0.reshape(n, -1), bv0.reshape(n, -1)], 1).astype(np.float32)
    s_post = np.concatenate([bp1.reshape(n, -1), bv1.reshape(n, -1)], 1).astype(np.float32)
    return s_pre, s_post
