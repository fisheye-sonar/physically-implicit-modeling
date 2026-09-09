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

from pim.metrics.zone_editability import build_edit_zones

N_OBJ, EF, K_ROLL, SEED = 2, 20, 15, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_REPO = Path(__file__).resolve().parents[3]
# The canonical edit set, always. New instance path preferred; legacy honoured until
# the Phase-2 data move lands.
_EVAL_NEW = _REPO / "datasets" / "discworld" / "dw-pn04" / "eval"
_EVAL_LEGACY = _REPO / "datasets" / "4_fixed_refl_inview"
DATA = _EVAL_NEW if _EVAL_NEW.exists() else _EVAL_LEGACY


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


def _to_basis(pos, vel, sim, basis_name):
    """World (x,y[,vx,vy]) -> the named basis. None/'cartesian' is a no-op."""
    if basis_name in (None, "cartesian"):
        return pos, vel
    from pim.environments.discworld.frustum import basis as fb

    return fb(pos, vel, sim, depth=basis_name)


def selection_path(data_dir: Path | None = None) -> Path:
    """An instance's FILTERED edit-case list, if it has one (``edits_selection.json``).

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
    d = Path(data_dir) if data_dir is not None else DATA
    return Path(d).parent / "edits_selection.json"


def bench_arrays(n: int = 192, target: str = "pos", basis_name: str = "cartesian",
                 data_dir: Path | None = None, select: np.ndarray | None = None,
                 use_selection: bool = True) -> dict:
    """The edit set's arrays and zones, model-free (factored out of ``load_bench``,
    2026-09-05, so a token model — ``token_bench`` — scores the SAME cases, targets
    and zones without a frame-space state).

    ``select`` (case indices into the edits split) replaces "the first ``n`` cases" —
    for subset benches (e.g. the dw-blink reappearance cases, 2026-09-07). The
    canonical bench is always ``select=None``.

    Uses ``pim.environments.discworld.loading``: ``clean_obs`` is RECONSTRUCTED from
    stored ids/reflectivities, not stored — reading the h5 directly gets a KeyError.
    On a blink instance the split's ``blink_visible`` schedule is handed to the zone
    construction so the reference worlds carry the same blackouts and markers.
    """
    from pim.environments.discworld.loading import load_edits

    dd = Path(data_dir) if data_dir is not None else DATA
    if select is None and use_selection:                 # the instance's filtered case list
        _sp = selection_path(dd)
        if _sp.exists():
            select = np.asarray(json.loads(_sp.read_text())["select"], dtype=int)[:n]
    # the edits split alone — its own config_json carries the sim config, so the 188 MB
    # test split is never decompressed just to read a dict (2026-09-07)
    b = load_edits(dd / "edits.h5", n_obj_keep=N_OBJ)
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
    bp, bv = _to_basis(pos[:, EF], vel[:, EF], sim, basis_name)
    y = bp.reshape(n, -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(n, -1)], axis=1)
    # The edit moves ONE object; everything else is a hold-the-rest constraint. Marking
    # too many dims would quietly turn a targeted edit into a whole-state overwrite.
    d_out = y.shape[1]
    cm = np.zeros((n, d_out), bool)
    cm[np.arange(n), 2 * eobj] = True
    cm[np.arange(n), 2 * eobj + 1] = True
    if target == "full":
        cm[np.arange(n), 2 * N_OBJ + 2 * eobj] = True
        cm[np.arange(n), 2 * N_OBJ + 2 * eobj + 1] = True
    out_dims = sorted({int(i) for i in np.where(cm.any(0))[0]})
    ef = int(getattr(b, "edit_frame", EF))
    assert ef == EF, f"edits split has edit_frame {ef}, every thread number assumes {EF}"
    return dict(obs=obs, pos=pos, vel=vel, edit_object=eobj, clean=clean, sim=sim,
                gt_roll=gt_roll, zones=zones, y=y, change_mask=cm, out_dims=out_dims, n=n,
                blink_visible=blink)


def load_bench(model, n: int = 192, target: str = "pos",
               basis_name: str = "cartesian", data_dir: Path | None = None,
               select: np.ndarray | None = None, use_selection: bool = True) -> Bench:
    """Warm ``model`` on the edits split and build the ground-truth zones (``bench_arrays``)."""
    a = bench_arrays(n, target, basis_name, data_dir, select=select, use_selection=use_selection)
    state = model.state_from_obs(torch.from_numpy(a["obs"][:, :EF]).float().to(DEV))
    return Bench(a["obs"], a["gt_roll"], a["zones"], torch.from_numpy(a["y"]).float().to(DEV),
                 torch.from_numpy(a["change_mask"]).to(DEV), a["out_dims"], state, a["n"],
                 pos=a["pos"], vel=a["vel"], edit_object=a["edit_object"], sim=a["sim"])


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
