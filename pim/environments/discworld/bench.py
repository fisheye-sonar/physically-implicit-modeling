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
import os
import tempfile
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
from pim.probes.mlp import CANONICAL_HIDDEN, fit_mlp

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
    # .resolve(): the cache key must not depend on how the path was SPELLED. A
    # relative data_dir (a pilot run from the repo root) and an absolute one
    # (master_eval, which resolves REPO) hashed to DIFFERENT keys, so every probe
    # was fitted and stored twice — and a 4-probe refit is a ~24 GB job that once
    # OOM-killed the desktop (2026-09-01).
    dd = (Path(data_dir) if data_dir is not None else DATA).resolve()
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
    # Disk-backed: the stack alone is 21.6-24.6 GB and the fits' temporaries must fit
    # beside it under the memory cap (see collect_residuals). Deleted after the fits.
    # ⛔ NOT the system tempdir: /tmp is tmpfs (RAM) on the lab box, which is how the
    # first "disk-backed" attempt filled 24.6 GB of RAM and hit a quota (2026-09-02).
    # The repo lives on nvme with terabytes free; .scratch/ is gitignored.
    _sdir = Path(__file__).resolve().parents[3] / ".scratch"
    _sdir.mkdir(exist_ok=True)
    _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=_sdir)
    _tmp.close()
    R = collect_residuals(model, obs, batch=64, memmap=_tmp.name)  # (NP, N, T, d)
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
    os.unlink(_tmp.name)
    if cache:
        store.store(fname, prov, out)
        if log:
            log(f"    probe cache WROTE {fname}")
    return out


def observation_probes(target: str = "full", n_seq: int = 30_000, split: str = "test",
                       family: str = "linear", basis_name: str = "cartesian",
                       span: int = 39, data_dir=None, cache_dir=None,
                       cache: bool = True, log=print, epochs: int | None = None) -> tuple:
    """The OBSERVATION floor: the canonical probes fitted to the causal observation
    history instead of a model's residual stream. No model is involved at all.

    Matched to ``fit_probes`` in every other respect — same corpus, same ``n_seq``, same
    SEEDed 80/20 split by sequence (identical permutation, so the held-out episodes are
    literally the same ones), same targets, same basis, same probe families — so the
    only difference between this row of Table 3 and a model row is the features.

    ``span`` matches the model's ``state_span`` so frames align one-for-one; the feature
    at frame t is obs[0..t] zero-padded to span, i.e. exactly what the model has consumed
    when its residual stream is read at t. Returns ``(probe, stats)`` — ONE probe, since
    there is no residual point to sweep.
    """
    import torch as _t

    from pim.probes.baselines import CausalHistory, fit_baseline_probe

    store = ProbeCache(cache_dir) if cache_dir is not None else PROBE_CACHE
    dd = (Path(data_dir) if data_dir is not None else DATA).resolve()
    # `epochs` enters the key only when set: the canonical fit (200 epochs on 30k) keeps its
    # existing keys; the 5x-corpus floor runs 50 epochs (>= 2x the canonical step count).
    extra = {} if epochs is None else {"epochs": int(epochs)}
    fname, prov = store.key(None, kind="observation", target=target, n_seq=int(n_seq),
                            split=split, family=family, basis=basis_name, seed=SEED,
                            span=int(span), data=str(dd), **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    obs-baseline cache HIT  {fname}")
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
    obs, y = obs[:, :span], y[:, :span]
    # THE SAME permutation fit_probes draws — the two floors and the model are compared
    # on identical held-out episodes, not merely on splits of the same size.
    perm = np.random.default_rng(SEED).permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    hist = CausalHistory(_t.from_numpy(obs).to(DEV))
    out = fit_baseline_probe(hist, _t.from_numpy(y).float().to(DEV), tr, te,
                             hidden=None if family == "linear" else CANONICAL_HIDDEN,
                             seed=SEED, log=log, **extra)
    if log:
        log(f"    obs baseline [{basis_name}/{family}]: R2 {out[1]['r2']:+.4f} "
            f"(in-sample {out[1]['r2_insample']:+.4f}, d_in {out[1]['d_in']})")
    if cache:
        store.store(fname, prov, out)
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
    """Free-run whose FIRST step is produced under a callable edit hook.

    A recurrent model carries its edited hiddens forward (``rollout_with_hook``); the
    transformers carry only the observation window, so for them the hook shapes one
    prediction and the rest of the rollout is recomputed unedited."""
    if hasattr(model, "rollout_with_hook"):
        return model.rollout_with_hook(state, hook, steps).cpu().numpy()
    pred = model.decode(state, edit=hook)
    out, s = [pred], model.advance(state, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


@torch.no_grad()
def unsteered(model, b: Bench) -> dict:
    """No intervention, through the IDENTICAL rollout path (state written back unchanged)."""
    roll = unsteered_rollout(model, b)
    c = score(model, b, roll)
    c["fidelity_ratio"] = 1.0
    return c


# ── rollouts (the editors' writes, without the scoring) ──────────────────────
#
# The *_arm functions score; these return the rollout itself, for qualitative panels
# (`notebooks/make_waterfalls.ipynb`). Each arm below is defined in terms of these, so
# an editor's write exists exactly once and a picture can never disagree with a score.


@torch.no_grad()
def free_rollout(model, obs: np.ndarray, teacher_force: int, steps: int) -> np.ndarray:
    """Teacher-force ``obs[:, :teacher_force]``, then free-run ``steps`` frames.

    No edit anywhere. Step 0 of the returned rollout is the model's prediction OF frame
    ``teacher_force`` — i.e. it aligns with ``clean_obs[:, teacher_force : +steps]``.
    """
    x = torch.from_numpy(np.asarray(obs)[:, :teacher_force]).float().to(DEV)
    s = model.state_from_obs(x)
    pred = model.decode(s)
    out = [pred]
    s = model.advance(s, pred)
    for _ in range(steps - 1):
        pr, s = model.predict_step(s)
        out.append(pr)
    return torch.stack(out, 1).cpu().numpy()


@torch.no_grad()
def unsteered_rollout(model, b: Bench) -> np.ndarray:
    """The no-intervention rollout, through the IDENTICAL path an edit takes."""
    ell = model.n_layers
    as_activations(model, ell)
    return model.rollout_with_edit(b.state, ell, model.flat_state(b.state),
                                   K_ROLL).cpu().numpy()


@torch.no_grad()
def pinv_rollout(model, b: Bench, probe, ell: int, alpha: float,
                 space: str = "zspace", dims: str = "all") -> np.ndarray:
    """PI's rollout at one residual point and step size."""
    as_activations(model, ell)
    h0 = model.flat_state(b.state)
    h = h0 + alpha * pinv_step(h0, b.tgt, probe, space=space, dims=dim_idx(dims))
    return model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()


@torch.no_grad()
def nanda_rollout(model, b: Bench, probe, ell: int, alpha: float,
                  dims: str = "all") -> np.ndarray:
    """ND's rollout at one residual point and step size."""
    idx = dim_idx(dims)
    rows = b.out_dims if idx is None else [d for d in b.out_dims if d in set(idx)]
    d = probe_direction(probe, rows)
    return _roll_hook(model, b.state, addition_hook(ell, d, alpha))


def grad_steer_rollout(model, b: Bench, probes: dict, start_layer: int, alpha: float,
                       n_steps: int = 100, beta: float = 0.2,
                       dims: str = "all") -> np.ndarray:
    """GS's rollout from ``start_layer`` and every residual point after it."""
    pts = {e: probes[e][0] for e in probes if e >= start_layer}
    cm = restrict_mask(b.change_mask, dims)
    specs = {}
    for e, pr in pts.items():
        as_activations(model, e)
        specs[e] = build_edit_spec(pr, model.flat_state(b.state), cm,
                                   b.tgt, beta=beta)
    hook = make_intervention_hook(pts, specs, start_layer, alpha=alpha, n_steps=n_steps)
    return _roll_hook(model, b.state, hook)


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


# ── the three workhorse arms ─────────────────────────────────────────────────


@torch.no_grad()
def nanda_arm(model, b: Bench, probe, ell: int, alphas,
              dims: str = "all") -> list[dict]:
    """ND at one residual point, α swept. Direction = the edited object's read-out rows."""
    recs = []
    for a in alphas:
        roll = nanda_rollout(model, b, probe, ell, a, dims=dims)
        recs.append({"editor": "ND", "point": ell, "alpha": float(a), "dims": dims,
                     "write_ratio": float(a), **score(model, b, roll)})
    return recs


@torch.no_grad()
def pinv_arm(model, b: Bench, probes: dict, alphas, space: str = "zspace",
             dims: str = "all") -> list[dict]:
    """PI at ONE residual point, tried at every point, α swept (α=1 = the exact jump).

    Both axes are load-bearing: 2026-08-21 measured 28× across points and ~50× across α.
    """
    idx = dim_idx(dims)
    recs = []
    for ell, (probe, _) in probes.items():
        as_activations(model, ell)
        h0 = model.flat_state(b.state)
        step = pinv_step(h0, b.tgt, probe, space=space, dims=idx)
        # the landing check is scored on the DRIVEN dims — see readout_error's docstring
        err0 = readout_error(h0, b.tgt, probe, dims=idx)
        for a in alphas:
            h = h0 + a * step
            roll = model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()
            recs.append({"editor": f"PI[{space}]", "point": ell, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float((a * step).norm(dim=1)
                                              .div(h0.norm(dim=1)).mean()),
                         "readout_err_before": err0,
                         "readout_err_after": readout_error(h, b.tgt, probe, dims=idx),
                         **score(model, b, roll)})
    return recs


def grad_steer_arm(model, b: Bench, probes: dict, start_layers, alphas,
                   n_steps: int = 100, beta: float = 0.2,
                   dims: str = "all") -> list[dict]:
    """GS from each start layer and EVERY point after it — Li's sequential schedule."""
    recs = []
    cm = restrict_mask(b.change_mask, dims)
    for ls in start_layers:
        pts = {e: probes[e][0] for e in probes if e >= ls}
        for a in alphas:
            specs = {}
            for e, pr in pts.items():
                as_activations(model, e)
                specs[e] = build_edit_spec(pr, model.flat_state(b.state),
                                           cm, b.tgt, beta=beta)
            rec: dict = {}
            hook = make_intervention_hook(pts, specs, ls, alpha=a, n_steps=n_steps,
                                          record=rec)
            roll = _roll_hook(model, b.state, hook)
            recs.append({"editor": f"GS@L{ls}", "point": ls, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0]
                             or [np.nan])),
                         **score(model, b, roll)})
    return recs
