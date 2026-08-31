"""The discworld editability bench: the canonical edit set, probes, and editor arms.

Ported 2026-08-31 from ``othello_arch/editability.py`` — the model-agnostic suite that
produced every discworld editability number since 2026-08-22 — now built ONLY from the
canonical parts: probes from ``pim.probes``, editors from ``pim.editors``, metrics from
``pim.metrics``. Nothing here re-derives a formula; this module is the *wiring* that
binds them to the dataset-4 edits split, plus the sweep loops the master notebook calls.

The bench is always the same 192 mid-sequence teleports from the canonical eval split
(``eval/edits.h5`` of the dw-pn04 instance = the old ``datasets/4_fixed_refl_inview``),
scored on ``pim.metrics.editability``'s ray zones — that constancy is what makes every
Edit Index in the project comparable, across models, bases, and probe corpora.

Editors follow the 2026-08-22 spec (single-point PI with an α sweep — both axes matter,
28× and ~50× respectively — Nanda addition per point, Li grad steering from every start
layer), with one change: **PI solves in z-space with the y-affine included**
(``pim.editors.pinv``, canonical since the 2026-08-31 affine fix). The pre-fix behaviour
is reproducible via ``space="legacy"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.nanda import addition_hook, probe_direction
from pim.editors.pinv import pinv_step, readout_error
from pim.metrics.editability import (
    build_edit_zones,
    edit_index_by_step,
    edit_scorecard,
    fidelity_ratio,
)
from pim.probes.base import collect_residuals
from pim.probes.cache import ProbeCache
from pim.probes.linear import fit_linear
from pim.probes.mlp import fit_mlp

N_OBJ, EF, K_ROLL, SEED = 2, 20, 15, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

_REPO = Path(__file__).resolve().parents[3]
# The canonical edit set, always. New instance path preferred; legacy honoured until
# the Phase-2 data move lands.
_EVAL_NEW = _REPO / "datasets" / "discworld" / "dw-pn04" / "eval"
_EVAL_LEGACY = _REPO / "datasets" / "4_fixed_refl_inview"
DATA = _EVAL_NEW if _EVAL_NEW.exists() else _EVAL_LEGACY
PROBE_CACHE = ProbeCache(_REPO / "runs" / "probe_cache" / "discworld")


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


def _to_basis(pos, vel, sim, basis_name):
    """World (x,y[,vx,vy]) -> the named basis. None/'cartesian' is a no-op."""
    if basis_name in (None, "cartesian"):
        return pos, vel
    from pim.environments.discworld.frustum import basis as fb

    return fb(pos, vel, sim, depth=basis_name)


def load_bench(model, n: int = 192, target: str = "pos",
               basis_name: str = "cartesian", data_dir: Path | None = None) -> Bench:
    """Warm ``model`` on the edits split and build the ground-truth zones.

    Uses ``pim.environments.discworld.loading``: ``clean_obs`` is RECONSTRUCTED from
    stored ids/reflectivities, not stored — reading the h5 directly gets a KeyError.
    """
    from pim.environments.discworld.loading import load_dataset

    dd = Path(data_dir) if data_dir is not None else DATA
    bundle = load_dataset(str(dd), n_obj_keep=N_OBJ)
    b = bundle.edits
    obs = b.obs[:n].astype(np.float32)
    pos = b.positions[:n, :, :N_OBJ, :].astype(np.float32)
    eobj = b.edit_object[:n].astype(int)
    clean = b.clean_obs[:n].astype(np.float32)
    with h5py.File(b.h5_path, "r") as f:
        vel = f["velocities"][:n, :, :N_OBJ, :].astype(np.float32)
    sim = bundle.test.config["dataset"]["sim"]  # `config` lives on `test`, not `edits`
    gt_roll = clean[:, EF: EF + K_ROLL, :]
    zones = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF],
                             pre_vel=vel[:, EF - 1], edit_object=eobj, sim=sim,
                             n_obj=N_OBJ, traj_pos=pos[:, EF: EF + K_ROLL],
                             gt_edited_traj=gt_roll)
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
    state = model.state_from_obs(torch.from_numpy(obs[:, :EF]).float().to(DEV))
    return Bench(obs, gt_roll, zones, torch.from_numpy(y).float().to(DEV),
                 torch.from_numpy(cm).to(DEV), out_dims, state, n)


# ── probes over residual points, cached ──────────────────────────────────────


def fit_probes(model, target: str = "pos", n_seq: int = 30_000, split: str = "test",
               family: str = "linear", log=print, basis_name: str = "cartesian",
               cache: bool = True, data_dir: Path | None = None,
               cache_dir: Path | None = None) -> dict:
    """One probe per residual point, held out BY SEQUENCE. ``family`` linear|mlp.

    ``data_dir`` supplies a LARGER corpus for probe fitting only — the bench stays the
    canonical edit set regardless. Cached with full provenance (``pim.probes.cache``).

    ``cache_dir`` is where the fitted probes LIVE. Canonical scoring passes the run's
    own ``runs/<topic>/<run>/probes/`` so every run dir is self-contained (re-tests
    never refit, and archiving a run carries its probes); the shared pool
    ``runs/probe_cache/discworld/`` is only the fallback for ad-hoc fits. The model
    fingerprint stays in every key either way, so a copied or overwritten checkpoint
    can never be served another model's probes.
    """
    store = ProbeCache(cache_dir) if cache_dir is not None else PROBE_CACHE
    dd = Path(data_dir) if data_dir is not None else DATA
    fname, prov = store.key(model, target=target, n_seq=int(n_seq), split=split,
                            family=family, basis=basis_name, seed=SEED,
                            data=str(dd))
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    probe cache HIT  {fname}  ({target}/{family}/{basis_name}/"
                    f"n={n_seq:,})")
            return hit
    with h5py.File(dd / f"{split}.h5", "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(dd / "dataset.json"))["sim"]
    bp, bv = _to_basis(pos, vel, sim, basis_name)
    y = bp.reshape(n_seq, bp.shape[1], -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(n_seq, bv.shape[1], -1)], axis=-1)
    # Transformer-L has a fixed block_size (39, learned absolute positions) and cannot
    # take a 40-frame episode; truncating here keeps both architectures on one path.
    span = getattr(model, "state_span", obs.shape[1])
    obs = obs[:, : min(obs.shape[1], span)]
    R = collect_residuals(model, obs, batch=64)  # (NP, N, T, d)
    y = y[:, : R.shape[2]]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    fit = fit_linear if family == "linear" else fit_mlp
    out = {}
    for ell in range(R.shape[0]):
        X = R[ell]
        p, s = fit(X[tr].reshape(-1, X.shape[-1]), y[tr].reshape(-1, y.shape[-1]),
                   X[te].reshape(-1, X.shape[-1]), y[te].reshape(-1, y.shape[-1]),
                   device=DEV, seed=SEED)
        out[ell] = (p, s)
        if log:
            log(f"    point {ell}: R2 {s['r2']:+.4f}  rmse {s['rmse']:.4f}")
    del R
    if cache:
        store.store(fname, prov, out)
        if log:
            log(f"    probe cache WROTE {fname}")
    return out


# ── scoring plumbing ─────────────────────────────────────────────────────────


def as_activations(model, ell: int):
    """Point a model's ``flat_state`` at residual point ``ell``."""
    if hasattr(model, "state_view"):
        model.state_view = "activations"
    model.probe_layer = ell
    return model


@torch.no_grad()
def score(model, b: Bench, roll: np.ndarray, uns_card: dict | None = None) -> dict:
    c = edit_scorecard(roll, b.zones, b.gt_roll)
    c["step0"] = float(edit_index_by_step(roll, b.zones, b.gt_roll)[0])
    if uns_card is not None:
        c["fidelity_ratio"] = fidelity_ratio(c, uns_card)
    return c


@torch.no_grad()
def _roll_hook(model, state, hook, steps: int = K_ROLL):
    """Free-run whose FIRST step is produced under a callable edit hook."""
    pred = model.decode(state, edit=hook)
    out, s = [pred], model.advance(state, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


@torch.no_grad()
def unsteered(model, b: Bench) -> dict:
    """No intervention, through the IDENTICAL rollout path (state written back unchanged)."""
    ell = model.n_layers  # last residual point
    as_activations(model, ell)
    roll = model.rollout_with_edit(b.state, ell, model.flat_state(b.state),
                                   K_ROLL).cpu().numpy()
    c = score(model, b, roll)
    c["fidelity_ratio"] = 1.0
    return c


# ── the three workhorse arms ─────────────────────────────────────────────────


@torch.no_grad()
def nanda_arm(model, b: Bench, probe, ell: int, alphas) -> list[dict]:
    """ND at one residual point, α swept. Direction = the edited object's read-out rows."""
    d = probe_direction(probe, b.out_dims)
    recs = []
    for a in alphas:
        roll = _roll_hook(model, b.state, addition_hook(ell, d, a))
        recs.append({"editor": "ND", "point": ell, "alpha": float(a),
                     "write_ratio": float(a), **score(model, b, roll)})
    return recs


@torch.no_grad()
def pinv_arm(model, b: Bench, probes: dict, alphas, space: str = "zspace") -> list[dict]:
    """PI at ONE residual point, tried at every point, α swept (α=1 = the exact jump).

    Both axes are load-bearing: 2026-08-21 measured 28× across points and ~50× across α.
    """
    recs = []
    for ell, (probe, _) in probes.items():
        as_activations(model, ell)
        h0 = model.flat_state(b.state)
        step = pinv_step(h0, b.tgt, probe, space=space)
        err0 = readout_error(h0, b.tgt, probe)
        for a in alphas:
            h = h0 + a * step
            roll = model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()
            recs.append({"editor": f"PI[{space}]", "point": ell, "alpha": float(a),
                         "write_ratio": float((a * step).norm(dim=1)
                                              .div(h0.norm(dim=1)).mean()),
                         "readout_err_before": err0,
                         "readout_err_after": readout_error(h, b.tgt, probe),
                         **score(model, b, roll)})
    return recs


def grad_steer_arm(model, b: Bench, probes: dict, start_layers, alphas,
                   n_steps: int = 100, beta: float = 0.2) -> list[dict]:
    """GS from each start layer and EVERY point after it — Li's sequential schedule."""
    recs = []
    for ls in start_layers:
        pts = {e: probes[e][0] for e in probes if e >= ls}
        for a in alphas:
            specs = {}
            for e, pr in pts.items():
                as_activations(model, e)
                specs[e] = build_edit_spec(pr, model.flat_state(b.state),
                                           b.change_mask, b.tgt, beta=beta)
            rec: dict = {}
            hook = make_intervention_hook(pts, specs, ls, alpha=a, n_steps=n_steps,
                                          record=rec)
            roll = _roll_hook(model, b.state, hook)
            recs.append({"editor": f"GS@L{ls}", "point": ls, "alpha": float(a),
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0]
                             or [np.nan])),
                         **score(model, b, roll)})
    return recs
