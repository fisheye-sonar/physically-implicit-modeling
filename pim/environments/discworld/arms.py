"""The discworld editor arms — probes over residual points, rollouts, oracles, and the
three workhorse arms (PI / ND / GS) — the counterpart of ``pim.environments.othello.arms``.

Built ONLY from the canonical parts: probes from ``pim.probes``, editors from
``pim.editors``, metrics from ``pim.metrics``; the edit set itself is ``bench.py``.
Nothing here re-derives a formula; this module is the *wiring* plus the sweep loops the
master notebook calls. (``bench.py`` held all of this until 2026-09-07.)

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
from pathlib import Path

import h5py
import numpy as np
import torch

from pim.editors.freeze_interpolation import freeze_time_rollout, frozen_frames
from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.nanda import addition_hook, probe_direction
from pim.editors.oracle_overwrite import overwrite_rollout
from pim.editors.pinv import pinv_step, readout_error, swap_class_logits
from pim.environments.discworld.bench import (
    DEV, EF, K_ROLL, N_OBJ, SEED, Bench, _to_basis, dim_idx, restrict_mask)
from pim.environments.discworld.grid_target import categorical_target
from pim.metrics.zone_editability import edit_scorecard, fidelity_ratio, object_constants
from pim.models.protocol import free_run
from pim.probes.base import FIT_BATCH, FIT_EPOCHS, collect_residuals
from pim.probes.cache import ProbeCache
from pim.probes.linear import fit_linear
from pim.probes.mlp import CANONICAL_HIDDEN, fit_mlp

__all__ = ["fit_probes", "observation_probes", "probe_recipe", "GRID_PROBE_RECIPE",
           "as_activations", "score", "unsteered",
           "free_rollout", "unsteered_rollout", "pinv_rollout", "nanda_rollout",
           "grad_steer_rollout", "counterfactual_history", "overwrite_oracle_rollout",
           "freeze_oracle_rollout", "oracle_arm", "nanda_arm", "pinv_arm", "grad_steer_arm",
           "fidelity_ratio", "Bench", "DEV", "EF", "K_ROLL", "N_OBJ", "SEED"]


def _require_cache_dir(cache_dir) -> Path:
    if cache_dir is None:
        raise ValueError("cache_dir is required: every fitted probe is persisted in a named "
                         "directory (the run's probes/ for canonical scoring, the experiment's "
                         "probes/ otherwise) — there is no shared pool")
    return Path(cache_dir)


# ── probes over residual points, cached ──────────────────────────────────────

# THE fit recipe of the grid-target probes (2026-09-08, canonicalised 2026-09-09): the
# instance's LARGE probe split, 200k sequences, 50 epochs — the large-corpus precedent
# (master_eval b3 observation floors, the probe-capacity sweep), 1.65× the gradient steps
# of the canonical 200 epochs over 30k sequences. The grid target is 98.4% "empty", so it
# was fitted on the wide corpus from the start; the canonical 30k recipe has NOT been run
# on it. Every existing grid probe carries this recipe in its cache key, and
# ``probe_recipe`` hands it to every caller (scorer, floors, waterfalls) so nobody can
# request — and silently trigger — a differently-keyed refit.
GRID_PROBE_RECIPE = {"probe_size": "250k", "n_seq": 200_000, "epochs": 50}


def probe_recipe(target: str, inst_root, n_seq: int = 30_000) -> dict:
    """``fit_probes`` / ``observation_probes`` keyword arguments that select an instance's
    probe corpus and fit length for ``target``: the canonical 120k corpus at ``n_seq`` and
    the default step count for the regression targets; ``GRID_PROBE_RECIPE`` for a grid.

    ``inst_root`` is the instance NAME (``"dw-8ray"``) or its directory (older callers).
    The corpus is named logically — ``{"probe": {"instance", "size"}}`` — and resolved to a
    file by ``pim.environments.layout`` at fit time (2026-09-10, layout v2)."""
    inst = inst_root if isinstance(inst_root, str) and "/" not in inst_root else Path(inst_root).name
    if categorical_target(target) is not None:
        r = GRID_PROBE_RECIPE
        return {"probe": {"instance": inst, "size": r["probe_size"]}, "n_seq": r["n_seq"],
                "epochs": r["epochs"]}
    return {"probe": {"instance": inst, "size": "120k"}, "n_seq": n_seq, "epochs": None}


def _probe_corpus(data_dir, probe, split: str) -> tuple[Path, Path, dict]:
    """(h5 file, manifest, cache-key fields) for a probe corpus request.

    ``probe`` = ``{"instance", "size"}`` (or an ``(instance, size)`` pair) names an instance
    corpus LOGICALLY; its cache key is ``layout.probe_key`` — path-free, so a dataset move
    cannot orphan a fitted probe. ``data_dir`` is the older form: an instance's probe
    directory maps onto the same logical key (so every existing call site produces the
    same key as the new form); any other directory (a pilot, a test's tmp corpus) is
    read as ``<dir>/<split>.h5`` and keyed by its resolved path, exactly as before.
    Neither given: the default instance's 120k corpus."""
    from pim.environments import layout

    if probe is not None:
        inst, size = (probe["instance"], probe["size"]) if isinstance(probe, dict) else probe
    elif data_dir is None:
        inst, size = layout.DEFAULT_INSTANCE["discworld"], "120k"
    else:
        lk = layout.legacy_probe_key(data_dir)
        if lk is None:
            dd = Path(data_dir).resolve()
            return dd / f"{split}.h5", dd / "dataset.json", {"split": split, "data": str(dd)}
        _, inst, size = lk
    data, key_split = layout.probe_key("discworld", inst, size)
    return (layout.probe_file("discworld", inst, size), layout.probe_manifest("discworld", inst, size),
            {"split": key_split, "data": data})


def _targets(target: str, pos: np.ndarray, vel: np.ndarray, sim: dict, basis_name: str):
    """(y, n_classes) for a probe target: regression values in the basis (``pos`` /
    ``full``), or the grid's (…, cells) integer labels (``grid-<nu>x<nd>``)."""
    grid = categorical_target(target)
    if grid is not None:
        y, _ = grid.label_frames(pos, sim)
        return y, grid.n_classes
    bp, bv = _to_basis(pos, vel, sim, basis_name)
    y = bp.reshape(*bp.shape[:-2], -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(*bv.shape[:-2], -1)], axis=-1)
    return y, None


def fit_probes(model, target: str = "pos", n_seq: int = 30_000, split: str = "test",
               family: str = "linear", log=print, basis_name: str = "cartesian",
               cache: bool = True, data_dir: Path | None = None,
               cache_dir: Path | None = None, encoder=None, encoder_tag: str | None = None,
               epochs: int | None = None, require_cached: bool = False,
               probe: dict | tuple | None = None) -> dict:
    """One probe per residual point, held out BY SEQUENCE. ``family`` linear|mlp.

    ``probe`` (2026-09-10) names the probe corpus logically — ``{"instance", "size"}``, as
    ``probe_recipe`` returns it; ``data_dir`` is the older path form (see ``_probe_corpus``).

    ``target`` is ``"pos"`` / ``"full"`` (regression in ``basis_name``) or a grid name such
    as ``"grid-16x8"`` (3-way classification per cell, 2026-09-09; ``basis_name`` must be
    ``"frustum"`` — the grid is defined there). Classification fits stream the residual
    stack from disk one point at a time (``fit_probe_stream`` over ``MemmapRows``): their
    recipe is the 200k-sequence large corpus (``GRID_PROBE_RECIPE``), where the dense path
    would need a 16 GB copy per point.

    ``epochs`` enters the cache key only when set, so every existing key stands; pass it
    through ``probe_recipe`` rather than by hand. ``require_cached`` raises on a cache miss
    instead of fitting — the scorer sets it for targets whose probes were fitted
    deliberately, outside the scoring loop, so a stale key can never start a multi-hour
    fit inside a notebook.

    ``encoder`` (2026-09-05): maps the float frames (N, T, R) to what the model consumes —
    token ids for a frames-as-tokens model. Applied after the span truncation, so the
    targets align exactly as for the regression models. ``encoder_tag`` names it in
    the cache key (omitted entirely when no encoder is used, so existing keys stand).

    ``data_dir`` supplies a LARGER corpus for probe fitting only — the bench stays the
    canonical edit set regardless. Cached with full provenance (``pim.probes.cache``).

    ``cache_dir`` is where the fitted probes LIVE and is REQUIRED (2026-09-07): canonical
    scoring passes the run's own ``runs/<topic>/<run>/probes/`` so every run dir is
    self-contained; an experiment passes its own ``experiments/<name>/probes/``. There is
    no shared pool — every fit is persisted somewhere named. The model fingerprint is in
    every key, so a copied or overwritten checkpoint can never be served another
    model's probes.
    """
    store = ProbeCache(_require_cache_dir(cache_dir))
    grid = categorical_target(target)
    if grid is not None and basis_name != "frustum":
        raise ValueError(f"{target} is defined in the frustum basis, got basis {basis_name!r}")
    # The corpus is keyed LOGICALLY (layout.probe_key) since 2026-09-10 — a path in the
    # key is how a relative/absolute spelling once fitted every probe twice (2026-09-01),
    # and how a dataset move would have orphaned all 394 cached probes.
    h5_path, manifest, keyf = _probe_corpus(data_dir, probe, split)
    extra = {} if encoder is None else {"encoder": encoder_tag or "custom"}
    if epochs is not None:
        extra["epochs"] = int(epochs)
    fname, prov = store.key(model, target=target, n_seq=int(n_seq), split=keyf["split"],
                            family=family, basis=basis_name, seed=SEED,
                            data=keyf["data"], **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    probe cache HIT  {fname}  ({target}/{family}/{basis_name}/"
                    f"n={n_seq:,})")
            return hit
    if require_cached:
        raise RuntimeError(f"no cached probes for {prov} in {store.dir} — this target's probes "
                           f"are fitted deliberately, not by the scorer (require_cached=True)")
    with h5py.File(h5_path, "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(manifest))["sim"]
    y, n_classes = _targets(target, pos, vel, sim, basis_name)
    # Transformer-L has a fixed block_size (39, learned absolute positions) and cannot
    # take a 40-frame episode; truncating here keeps both architectures on one path.
    span = getattr(model, "state_span", obs.shape[1])
    obs = obs[:, : min(obs.shape[1], span)]
    if encoder is not None:
        obs = encoder(obs)                      # e.g. (N, T) token ids
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    # Disk-backed: the stack alone is 21.6-24.6 GB and the fits' temporaries must fit
    # beside it under the memory cap (see collect_residuals). Deleted after the fits.
    # ⛔ NOT the system tempdir: /tmp is tmpfs (RAM) on the lab box, which is how the
    # first "disk-backed" attempt filled 24.6 GB of RAM and hit a quota (2026-09-02).
    # The repo lives on nvme with terabytes free; .scratch/ is gitignored.
    _sdir = Path(__file__).resolve().parents[3] / ".scratch"
    _sdir.mkdir(exist_ok=True)
    out = {}
    if n_classes is not None:
        # Classification: one residual point at a time (200k seq × 39 × 512 × 4 B = 16 GB
        # per point), streamed by sequence block through the shared fitter.
        from pim.probes.baselines import MemmapRows, fit_probe_stream

        y_t = torch.from_numpy(y[:, : obs.shape[1]]).to(DEV)
        for ell in range(model.n_layers + 1):
            _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=_sdir)
            _tmp.close()
            try:
                R = collect_residuals(model, obs, batch=64, memmap=_tmp.name, points=[ell])
                p, s = fit_probe_stream(MemmapRows(R[0], device=DEV), y_t, tr, te,
                                        hidden=None if family == "linear" else CANONICAL_HIDDEN,
                                        n_classes=n_classes, seed=SEED,
                                        epochs=epochs or FIT_EPOCHS, batch=FIT_BATCH, log=None)
                del R
            finally:
                os.unlink(_tmp.name)
            out[ell] = (p, s)
            if log:
                log(f"    point {ell}: err {s['error_rate']:.3f}%  "
                    f"(majority {s['majority_class_error_rate']:.3f}%)")
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False, dir=_sdir)
        _tmp.close()
        try:
            R = collect_residuals(model, obs, batch=64, memmap=_tmp.name)  # (NP, N, T, d)
            y = y[:, : R.shape[2]]
            fit = fit_linear if family == "linear" else fit_mlp
            kw = {} if epochs is None else {"epochs": int(epochs)}
            for ell in range(R.shape[0]):
                X = R[ell]
                p, s = fit(X[tr].reshape(-1, X.shape[-1]), y[tr].reshape(-1, y.shape[-1]),
                           X[te].reshape(-1, X.shape[-1]), y[te].reshape(-1, y.shape[-1]),
                           device=DEV, seed=SEED, **kw)
                out[ell] = (p, s)
                if log:
                    log(f"    point {ell}: R2 {s['r2']:+.4f}  rmse {s['rmse']:.4f}")
            del R
        finally:
            os.unlink(_tmp.name)          # a failed fit must not leave 20+ GB on the nvme
    if cache:
        store.store(fname, prov, out)
        if log:
            log(f"    probe cache WROTE {fname}")
    return out


def observation_probes(target: str = "full", n_seq: int = 30_000, split: str = "test",
                       family: str = "linear", basis_name: str = "cartesian",
                       span: int = 39, data_dir=None, cache_dir=None,
                       cache: bool = True, log=print, epochs: int | None = None,
                       align: str = "left", require_cached: bool = False,
                       probe: dict | tuple | None = None) -> tuple:
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
    from pim.probes.baselines import CausalHistory, fit_baseline_probe

    store = ProbeCache(_require_cache_dir(cache_dir))
    h5_path, manifest, keyf = _probe_corpus(data_dir, probe, split)   # logical key, see fit_probes
    # `epochs` enters the key only when set: the canonical fit (200 epochs on 30k) keeps its
    # existing keys; the 5x-corpus floor runs 50 epochs (>= 2x the canonical step count).
    extra = {} if epochs is None else {"epochs": int(epochs)}
    if align != "left":                     # existing (left-aligned) keys stay as they are
        extra["align"] = align
    fname, prov = store.key(None, kind="observation", target=target, n_seq=int(n_seq),
                            split=keyf["split"], family=family, basis=basis_name, seed=SEED,
                            span=int(span), data=keyf["data"], **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    obs-baseline cache HIT  {fname}")
            return hit
    if require_cached:
        raise RuntimeError(f"no cached observation probe for {prov} in {store.dir} "
                           f"(require_cached=True — see fit_probes)")
    with h5py.File(h5_path, "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(manifest))["sim"]
    y, n_classes = _targets(target, pos, vel, sim, basis_name)   # grid → labels, 3 classes
    obs, y = obs[:, :span], y[:, :span]
    # THE SAME permutation fit_probes draws — the two floors and the model are compared
    # on identical held-out episodes, not merely on splits of the same size.
    perm = np.random.default_rng(SEED).permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
    hist = CausalHistory(torch.from_numpy(obs).to(DEV), align=align)
    y_t = torch.from_numpy(y).to(DEV)
    out = fit_baseline_probe(hist, y_t if n_classes else y_t.float(), tr, te,
                             hidden=None if family == "linear" else CANONICAL_HIDDEN,
                             n_classes=n_classes,
                             seed=SEED, log=log, **{k: v for k, v in extra.items() if k != "align"})
    if log:
        from pim.metrics.decodability import probe_skill_from_stats
        log(f"    obs baseline [{basis_name}/{family}]: skill {probe_skill_from_stats(out[1]):+.4f} "
            f"(d_in {out[1]['d_in']})")
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
    return free_run(model, pred, model.advance(state, pred), steps).cpu().numpy()


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
    return free_run(model, pred, model.advance(s, pred), steps).cpu().numpy()


@torch.no_grad()
def unsteered_rollout(model, b: Bench) -> np.ndarray:
    """The no-intervention rollout, through the IDENTICAL path an edit takes."""
    ell = model.n_layers
    as_activations(model, ell)
    return model.rollout_with_edit(b.state, ell, model.flat_state(b.state),
                                   K_ROLL).cpu().numpy()


def _check_dims(b: Bench, dims: str) -> None:
    """A categorical target has no position / velocity read-outs to restrict to."""
    if b.kind == "classification" and dim_idx(dims) is not None:
        raise ValueError(f"dims={dims!r} is a regression dim set; a {b.kind} bench takes 'all'")


@torch.no_grad()
def pinv_target(probe, h0: torch.Tensor, b: Bench) -> torch.Tensor:
    """What PI asks the LINEAR probe to read after the write, in the probe's output units.

    Regression: the bench's target values. Classification (the grid target): the probe's
    OWN current read-out with empty ↔ the object's class swapped at the old cell and at
    the new cell — Othello's tile swap, on two cells — flattened to (B, cells·classes)."""
    if b.kind != "classification":
        return b.tgt
    zero = torch.zeros_like(b.cells["cls"])
    lg = swap_class_logits(probe(h0), b.cells["A"], zero, b.cells["cls"])
    lg = swap_class_logits(lg, b.cells["B"], zero, b.cells["cls"])
    return lg.reshape(h0.shape[0], -1)


@torch.no_grad()
def readout_landed(h: torch.Tensor, probe, b: Bench) -> float:
    """Classification landing check: the fraction of cases whose probe read-out at ``h``
    labels the old cell empty AND the new cell with the object — the categorical
    counterpart of ``readout_error``."""
    lab = probe(h).argmax(-1)
    ar = torch.arange(h.shape[0], device=h.device)
    ok = (lab[ar, b.cells["A"]] == 0) & (lab[ar, b.cells["B"]] == b.cells["cls"])
    return float(ok.float().mean())


@torch.no_grad()
def pinv_rollout(model, b: Bench, probe, ell: int, alpha: float,
                 space: str = "zspace", dims: str = "all") -> np.ndarray:
    """PI's rollout at one residual point and step size."""
    _check_dims(b, dims)
    as_activations(model, ell)
    h0 = model.flat_state(b.state)
    h = h0 + alpha * pinv_step(h0, pinv_target(probe, h0, b), probe, space=space,
                               dims=dim_idx(dims))
    return model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()


@torch.no_grad()
def nanda_rollout(model, b: Bench, probe, ell: int, alpha: float,
                  dims: str = "all") -> np.ndarray:
    """ND's rollout at one residual point and step size.

    Regression: one shared direction, the edited object's read-out rows summed (the
    reason ND is not reported on that target — see the registry). Classification: the
    Othello form, per case — the probe row of (new cell, class) minus the row of
    (old cell, class), the "move the object" direction (target − current contrast)."""
    _check_dims(b, dims)
    if b.kind == "classification":
        C = probe.n_classes
        d = probe_direction(probe, b.cells["B"] * C + b.cells["cls"],
                            subtract_rows=b.cells["A"] * C + b.cells["cls"], per_sample=True)
    else:
        idx = dim_idx(dims)
        rows = b.out_dims if idx is None else [d for d in b.out_dims if d in set(idx)]
        d = probe_direction(probe, rows)
    return _roll_hook(model, b.state, addition_hook(ell, d, alpha))


def grad_steer_rollout(model, b: Bench, probes: dict, start_layer: int, alpha: float,
                       n_steps: int = 100, beta: float = 0.2,
                       dims: str = "all", record: dict | None = None) -> np.ndarray:
    """GS's rollout from ``start_layer`` and every residual point after it.

    ``record`` (optional dict) receives the hook's per-point diagnostics (the arm reads
    ``delta_norm``/``x_norm`` from it for ``write_ratio``). On a classification bench the
    spec is Li's own: cross-entropy toward the bench's labels on the changed cells, the
    probe's current labels held elsewhere (``build_edit_spec`` branches on the probe)."""
    _check_dims(b, dims)
    pts = {e: probes[e][0] for e in probes if e >= start_layer}
    cm = restrict_mask(b.change_mask, dims)
    specs = {}
    for e, pr in pts.items():
        as_activations(model, e)
        specs[e] = build_edit_spec(pr, model.flat_state(b.state), cm,
                                   b.tgt, beta=beta)
    hook = make_intervention_hook(pts, specs, start_layer, alpha=alpha, n_steps=n_steps,
                                  record=record)
    return _roll_hook(model, b.state, hook)


# ── the two ORACLE editors on the bench (kept in the back pocket, 2026-09-07) ─


def counterfactual_history(b: Bench, noise_matched: bool = True, seed: int = 0) -> np.ndarray:
    """(N, EF, R) the frames 0..EF-1 the model would have SEEN had the edited world held
    all along: the edited object displaced by its teleport vector throughout the window,
    the other object on its true trajectory. Rendered through the instance's own renderer
    config, with its observation noise when ``noise_matched``."""
    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import sim_config_from

    assert b.pos is not None and b.vel is not None and b.sim is not None, "load_bench() fills these"
    cfg = sim_config_from(b.sim, N_OBJ)
    if noise_matched:
        cfg = type(cfg)(**{**cfg.__dict__, "obs_noise_std": float(b.sim["obs_noise_std"])})
    rad, refl = object_constants(b.sim, N_OBJ)
    idx, k = np.arange(b.n), b.edit_object.astype(int)
    dt = float(b.sim["dt"])
    # teleport vector = post-edit position at EF minus where the object would have been
    delta = b.pos[idx, EF, k] - (b.pos[idx, EF - 1, k] + b.vel[idx, EF - 1, k] * dt)
    hist = b.pos[:, :EF].copy()
    hist[idx, :, k] += delta[:, None, :]
    rng = np.random.default_rng(seed)
    out = np.zeros((b.n, EF, b.obs.shape[-1]), np.float32)
    for i in range(b.n):
        for t in range(EF):
            out[i, t] = render_frame(hist[i, t], rad, refl, cfg, rng=rng)[2]
    return out


@torch.no_grad()
def overwrite_oracle_rollout(model, b: Bench, noise_matched: bool = True) -> np.ndarray:
    """ORACLE 1: the state the model would carry had it SEEN the edited world for the
    whole window (``counterfactual_history``), then a free-run. Step 0 predicts frame EF,
    aligned with every other arm — it writes every dimension of the carried state, which
    is what makes it the ceiling on "the dynamics can carry the edited world at all".
    (A one-frame overwrite — the post-edit frame appended to the pre-edit window — is NOT
    accepted by a window model: EI +0.04 on L-dw-20m, 2026-09-07.)"""
    cf = torch.from_numpy(counterfactual_history(b, noise_matched)).float().to(DEV)
    return overwrite_rollout(model, cf, K_ROLL).cpu().numpy()


@torch.no_grad()
def freeze_oracle_rollout(model, b: Bench, n_frames: int = 8,
                          noise_matched: bool = True) -> np.ndarray:
    """ORACLE 2: teacher-force ``n_frames`` rendered frames in which the edited object
    glides pre -> target with time frozen (the other object holds at its frame-EF
    position), then free-run. Step 0 predicts frame EF, aligned with every other arm.
    ``noise_matched`` renders the frozen frames with the instance's observation noise."""
    assert b.pos is not None and b.sim is not None, "load_bench() builds the oracle fields"
    _, refl = object_constants(b.sim, N_OBJ)
    noise = float(b.sim["obs_noise_std"]) if noise_matched else 0.0
    frames = np.stack([
        frozen_frames(b.sim, b.pos[i, EF - 1], b.pos[i, EF], int(b.edit_object[i]), n_frames,
                      reflectivities=refl, obs_noise_std=noise, seed=i)
        for i in range(b.n)])
    return freeze_time_rollout(model, b.state,
                               torch.from_numpy(frames).float().to(DEV), K_ROLL).cpu().numpy()


@torch.no_grad()
def oracle_arm(model, b: Bench, n_freeze: int = 8) -> list[dict]:
    """Both oracle editors, scored like any arm. They exist to defend the Edit Index: if
    the measure can score well under a write that provably carries the edited world, a
    workhorse editor at the unedited floor is a fact about the model, not the measure."""
    u = unsteered(model, b)
    ov = score(model, b, overwrite_oracle_rollout(model, b), u)
    fz = score(model, b, freeze_oracle_rollout(model, b, n_freeze), u)
    return [{"editor": "ORACLE-overwrite", **{k: v for k, v in ov.items() if np.isscalar(v)}},
            {"editor": f"ORACLE-freeze[N={n_freeze}]", "n_freeze": n_freeze,
             **{k: v for k, v in fz.items() if np.isscalar(v)}}]


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
    _check_dims(b, dims)
    idx = dim_idx(dims)
    recs = []
    for ell, (probe, _) in probes.items():
        as_activations(model, ell)
        h0 = model.flat_state(b.state)
        tgt = pinv_target(probe, h0, b)
        step = pinv_step(h0, tgt, probe, space=space, dims=idx)
        # the landing check is scored on the DRIVEN dims — see readout_error's docstring;
        # on a categorical target it is the fraction of cases whose labels moved
        if b.kind == "classification":
            landing = lambda h: {"readout_landed": readout_landed(h, probe, b)}  # noqa: E731
            before = {"readout_landed_before": readout_landed(h0, probe, b)}
        else:
            landing = lambda h: {"readout_err_after": readout_error(h, tgt, probe, dims=idx)}  # noqa: E731
            before = {"readout_err_before": readout_error(h0, tgt, probe, dims=idx)}
        for a in alphas:
            h = h0 + a * step
            roll = pinv_rollout(model, b, probe, ell, a, space=space, dims=dims)  # THE write
            recs.append({"editor": f"PI[{space}]", "point": ell, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float((a * step).norm(dim=1)
                                              .div(h0.norm(dim=1)).mean()),
                         **before, **landing(h),
                         **score(model, b, roll)})
    return recs


def grad_steer_arm(model, b: Bench, probes: dict, start_layers, alphas,
                   n_steps: int = 100, beta: float = 0.2,
                   dims: str = "all") -> list[dict]:
    """GS from each start layer and EVERY point after it — Li's sequential schedule."""
    recs = []
    for ls in start_layers:
        for a in alphas:
            rec: dict = {}
            roll = grad_steer_rollout(model, b, probes, ls, a, n_steps=n_steps, beta=beta,
                                      dims=dims, record=rec)                  # THE write
            recs.append({"editor": f"GS@L{ls}", "point": ls, "alpha": float(a),
                         "dims": dims,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0]
                             or [np.nan])),
                         **score(model, b, roll)})
    return recs
