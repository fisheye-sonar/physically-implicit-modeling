"""The editability suite, run against ANY discworld model that exposes the standard surface.

Written model-agnostic on purpose, so the same call produces `W16` (ours) and `A_pilot` (theirs)
numbers on the identical edit set — the comparison is the point, and a second implementation would
make it untrustworthy. The four names a model must supply are `state_from_obs`, `flat_state`,
`decode(state, edit=)` and `rollout_with_edit`; `pim.world_models.transformer.TransformerModel`
has them natively and `othello_arch.model.OthelloArchDiscworld` gained them as its bridge.

**Nothing here re-derives a metric or an editor.** Probes come from
`othello_gpt/othello_probe.fit_probe`, the gradient editor from that module's
`build_edit_spec` / `make_intervention_hook`, the pseudoinverse from
`pim.editors.probe_steering.inject_state`, and every score from `scripts/editability_metrics`
(`build_edit_zones`, `edit_scorecard`, `fidelity_ratio`, `edit_index_by_step`).

Probes
------
Two target sets, both from `othello_gpt/pipeline.TARGETS`, at **every** residual point:
  * `pos`  — 4 dims, object positions. The read-out the edit actually drives.
  * `full` — 8 dims, positions **and** velocities, so the probe carries the whole world state.
Linear and MLP families, held out **by sequence** (`harness/ANALYSIS.md` §2).

Editors — all three, as specified 2026-08-22
--------------------------------------------
  1. **Nanda direction addition** — move along the probe weight ROWS for the target outputs only,
     with a magnitude sweep. `x <- x + alpha * w_d / x_std`, unit-normed.
  2. **Pseudoinverse injection at ONE residual point**, tried at every point, with a step-size
     sweep. 2026-08-21 established both that single-point matters (28x on Othello) and that alpha
     matters (~50x), so neither axis may be collapsed.
  3. **MLP gradient steering @L_s** — Li et al.'s method: descend on the activation through a
     frozen MLP probe at `L_s` **and every point after it**, alternating write and compute.

Reported
--------
Edit Index for the **unedited** and **edited** outcome (never a gain alone — 2026-08-22: the gain
rises with model quality even when absolute editability is flat), plus `fidelity_ratio`, the zone
RMSEs, and the read-out error before/after so an inert write is distinguishable from a landed
write the model ignores.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_probe as op  # noqa: E402
from pim.editors.probe_steering import inject_state  # noqa: E402
from editability_metrics import (  # noqa: E402
    build_edit_zones,
    edit_index_by_step,
    edit_scorecard,
    fidelity_ratio,
)

N_OBJ, EF, K_ROLL, SEED = 2, 20, 15, 0
DATA = _REPO / "datasets" / "4_fixed_refl_inview"   # the canonical edit set, always
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Bench:
    """The dataset-4 edits split: what every editability number in this repo is scored on."""

    obs: np.ndarray          # (N, T, R) noisy observations
    gt_roll: np.ndarray      # (N, K, R) clean post-edit ground truth from the edit frame
    zones: object            # target / ghost / differing ray masks
    tgt: torch.Tensor        # (N, d_out) the probe target the edit asks for
    change_mask: torch.Tensor  # (N, d_out) bool — the EDITED object's dims only
    out_dims: list[int]      # the read-out rows the edit is allowed to move
    state: object            # the model state warmed on obs[:, :EF]
    n: int


def _to_basis(pos, vel, sim, basis_name):
    """World (x,y[,vx,vy]) -> the named basis. `basis_name` None/'cartesian' is a no-op."""
    if basis_name in (None, "cartesian"):
        return pos, vel
    from pim.simulator.frustum import basis as fb
    return fb(pos, vel, sim, depth=basis_name)


def load_bench(model, n: int = 192, target: str = "pos", basis_name: str = "cartesian") -> Bench:
    """Warm `model` on the dataset-4 edits split and build the ground-truth zones.

    Uses `pim.world_models.load_dataset`, not raw h5: `clean_obs` is **reconstructed** from the
    stored seeds rather than stored, so reading the file directly gets a KeyError.
    """
    from pim.world_models import load_dataset

    bundle = load_dataset(str(DATA), n_obj_keep=N_OBJ)
    b = bundle.edits
    obs = b.obs[:n].astype(np.float32)
    pos = b.positions[:n, :, :N_OBJ, :].astype(np.float32)
    eobj = b.edit_object[:n].astype(int)
    clean = b.clean_obs[:n].astype(np.float32)
    with h5py.File(b.h5_path, "r") as f:
        vel = f["velocities"][:n, :, :N_OBJ, :].astype(np.float32)
    sim = bundle.test.config["dataset"]["sim"]   # `config` lives on `test`, not on `edits`
    gt_roll = clean[:, EF : EF + K_ROLL, :]
    zones = build_edit_zones(pre_pos=pos[:, EF - 1], tgt_pos=pos[:, EF], pre_vel=vel[:, EF - 1],
                             edit_object=eobj, sim=sim, n_obj=N_OBJ,
                             traj_pos=pos[:, EF : EF + K_ROLL], gt_edited_traj=gt_roll)
    # ⛔ The ZONES stay in world space — they are ray masks over the observation and do not depend
    #    on how we choose to coordinatise the state. Only the PROBE TARGET changes basis, so the
    #    Edit Index remains directly comparable across bases.
    bp, bv = _to_basis(pos[:, EF], vel[:, EF], sim, basis_name)
    y = bp.reshape(n, -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(n, -1)], axis=1)
    # The edit moves ONE object. Its dims are (2*obj, 2*obj+1) for position, and for the `full`
    # target the matching velocity dims too. Everything else is a hold-the-rest constraint, so
    # marking too many dims would quietly turn a targeted edit into a whole-state overwrite.
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


PROBE_CACHE = _REPO / "runs" / "othello_arch" / "probe_cache"


def _fingerprint(model) -> str:
    """12 hex chars over every parameter — a model's identity, cheaply.

    Same construction as `ours_on_othello/evaluate.py:fingerprint`, deliberately: a cache key
    that omits the weights is how the random-init control was once served the trained model's
    probes (2026-08-21), and both then reported identical error.
    """
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _probe_cache_key(model, *, target, n_seq, split, hidden, basis_name,
                     data_dir=None) -> tuple[str, dict]:
    """(filename, provenance). Every input that changes the fitted probe is in the provenance."""
    prov = {
        "model": _fingerprint(model),
        "target": target,
        "n_seq": int(n_seq),
        "split": split,
        "hidden": "linear" if hidden is None else int(hidden),
        "basis": basis_name,
        "seed": int(SEED),
        "data": str(data_dir if data_dir is not None else DATA),
        "span": int(getattr(model, "state_span", -1)),
        "v": 1,   # bump to invalidate every cached probe after a fitting change
    }
    h = hashlib.blake2b(repr(sorted(prov.items())).encode(), digest_size=8).hexdigest()
    return f"probes_{h}.pt", prov


def fit_probes(model, target: str = "pos", n_seq: int = 1500, split: str = "test",
               hidden: int | None = 512, log=print, basis_name: str = "cartesian",
               cache: bool = True, data_dir: Path | None = None) -> dict:
    """One probe per residual point, held out BY SEQUENCE. `target` in {'pos','full'}.

    Cached on disk under `runs/othello_arch/probe_cache/`, keyed on a hash of the model weights
    *and* every fitting argument. A hit is verified against the stored provenance before it is
    returned, so a key collision or a hand-edited cache file raises instead of silently serving
    the wrong probe. Pass `cache=False` to force a refit.
    """
    dd = Path(data_dir) if data_dir is not None else DATA
    fname, prov = _probe_cache_key(model, target=target, n_seq=n_seq, split=split,
                                   hidden=hidden, basis_name=basis_name, data_dir=dd)
    fpath = PROBE_CACHE / fname
    if cache and fpath.exists():
        blob = torch.load(fpath, map_location=DEV, weights_only=False)
        if blob.get("provenance") != prov:
            raise RuntimeError(
                f"probe cache provenance mismatch at {fpath}\n"
                f"  on disk: {blob.get('provenance')}\n  wanted : {prov}\n"
                f"Delete the file to refit. This should be unreachable — the filename is a hash "
                f"of exactly this dict — so reaching it means the cache was tampered with.")
        if log:
            log(f"    probe cache HIT  {fname}  ({target}/{prov['hidden']}/{basis_name}/"
                f"n={n_seq:,})")
        return blob["probes"]
    with h5py.File(dd / f"{split}.h5", "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    import json as _json

    _sim = _json.load(open(dd / "dataset.json"))["sim"]
    bp, bv = _to_basis(pos, vel, _sim, basis_name)
    y = bp.reshape(n_seq, bp.shape[1], -1)
    if target == "full":
        y = np.concatenate([y, bv.reshape(n_seq, bv.shape[1], -1)], axis=-1)
    # Their architecture has a fixed `block_size` (39) and learned absolute positions, so it
    # cannot be handed a 40-frame episode. Ours (`state_span` 61) is unaffected. Truncating here
    # rather than in the caller keeps the two models on one code path.
    span = getattr(model, "state_span", obs.shape[1])
    obs = obs[:, : min(obs.shape[1], span)]
    R = op.collect_residuals(model, obs, batch=64)              # (NP, N, T, d)
    T = R.shape[2]
    y = y[:, :T]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_seq)
    tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq) :]
    out = {}
    for ell in range(R.shape[0]):
        X = R[ell]
        p, s = op.fit_probe(X[tr].reshape(-1, X.shape[-1]), y[tr].reshape(-1, y.shape[-1]),
                            X[te].reshape(-1, X.shape[-1]), y[te].reshape(-1, y.shape[-1]),
                            hidden=hidden, device=DEV, seed=SEED)
        out[ell] = (p, s)
        if log:
            log(f"    point {ell}: R2 {s['r2']:+.4f}  rmse {s['rmse']:.4f}")
    del R
    if cache:
        PROBE_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = fpath.with_suffix(".pt.partial")
        torch.save({"provenance": prov, "probes": out}, tmp)
        tmp.replace(fpath)          # atomic: a killed run never leaves a half-written cache hit
        if log:
            log(f"    probe cache WROTE {fname}")
    return out


class ProbeSanityError(AssertionError):
    """An MLP probe scored WORSE than a linear one on held-out data."""


def check_probe_sanity(lin: dict, mlp: dict, *, tol: float = 0.01, strict: bool = True,
                       label: str = "", log=print) -> dict:
    """Tripwire: a held-out MLP probe must not be beaten by a linear probe.

    An MLP with a hidden layer can represent the linear map exactly, so on held-out data it
    should never score *below* a linear probe on the same features and targets. When it does,
    the MLP is fitting the probe training set rather than the model's representation, and every
    decodability number from that fit is meaningless.

    This is not hypothetical. On 2026-08-22 MLP probes on 1,500 sequences reported in-sample
    velocity R^2 of 0.954-0.959 against held-out -0.073 and -0.090 — 262k probe parameters on
    48k rows. The numbers looked like "velocity is barely decodable" and were quoted as such;
    refitting on 10k sequences moved held-out vy R^2 to 0.83. The tripwire below is the check
    that would have caught it at the point of fitting instead of two findings later.

    Returns a report dict (always), and raises `ProbeSanityError` when `strict`.
    """
    rows, bad = [], []
    for ell in sorted(set(lin) & set(mlp)):
        sl, sm = lin[ell][1], mlp[ell][1]
        r_lin, r_mlp = float(sl["r2"]), float(sm["r2"])
        gap_mlp = float(sm.get("r2_insample", np.nan)) - r_mlp
        gap_lin = float(sl.get("r2_insample", np.nan)) - r_lin
        row = {"point": ell, "r2_linear": r_lin, "r2_mlp": r_mlp,
               "mlp_minus_linear": r_mlp - r_lin,
               "insample_gap_mlp": gap_mlp, "insample_gap_linear": gap_lin}
        rows.append(row)
        if r_mlp < r_lin - tol:
            bad.append(row)
    report = {"label": label, "tol": tol, "rows": rows, "n_violations": len(bad)}
    if log:
        worst = max(rows, key=lambda r: r["insample_gap_mlp"]) if rows else None
        if worst is not None:
            log(f"    probe sanity{' [' + label + ']' if label else ''}: "
                f"{len(bad)}/{len(rows)} points where MLP < linear; "
                f"worst MLP in-sample gap {worst['insample_gap_mlp']:+.4f} @ point {worst['point']}")
    if bad and strict:
        det = "\n".join(f"      point {r['point']}: linear {r['r2_linear']:+.4f} > "
                         f"MLP {r['r2_mlp']:+.4f} (by {-r['mlp_minus_linear']:.4f}), "
                         f"MLP in-sample gap {r['insample_gap_mlp']:+.4f}" for r in bad)
        raise ProbeSanityError(
            f"MLP probe beaten by linear probe at {len(bad)} residual point(s)"
            f"{' for ' + label if label else ''} — the MLP is fitting the probe training set, "
            f"not the representation. Refit with more probe sequences (>=10k) before trusting "
            f"any decodability number from it.\n{det}")
    return report


def as_activations(model, ell: int):
    """Point a model's `flat_state` at residual point `ell`.

    `TransformerModel` defaults to `state_view="obs_window"`, whose `flat_state` returns the
    FLATTENED OBSERVATION BUFFER (61x128 = 7808 dims), not an activation — writing that into a
    residual point fails loudly, which is lucky; silently probing it would be worse.
    `OthelloArchDiscworld` has only the activation view, so the attribute is set defensively.
    """
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
def n_points(model) -> int:
    return int(getattr(model, "n_layers", None) or model.cfg.n_layers) + 1


@torch.no_grad()
def unsteered(model, b: Bench) -> dict:
    """No intervention, through the identical rollout path — writing back the state unchanged."""
    ell = n_points(model) - 1
    as_activations(model, ell)
    roll = model.rollout_with_edit(b.state, ell, model.flat_state(b.state), K_ROLL).cpu().numpy()
    c = score(model, b, roll)
    c["fidelity_ratio"] = 1.0
    return c


# ── the three editors ─────────────────────────────────────────────────────────


@torch.no_grad()
def nanda_addition(model, b: Bench, probe, ell: int, alphas, out_dims=None) -> list[dict]:
    """1. Move along the probe's weight ROWS for the target outputs only, magnitude swept.

    `out_dims` selects which read-out rows to move (default: all of them). The direction is taken
    in RAW activation space — the probe standardises its input, so `w / x_std` is the true
    raw-space gradient of that output, not `w`.
    """
    W = probe.net.weight.detach() if hasattr(probe.net, "weight") else probe.net[-1].weight.detach()
    rows = W if out_dims is None else W[list(out_dims)]   # only the EDITED object's read-out rows
    d = (rows / probe.x_std).sum(0)
    d = d / d.norm()
    recs = []
    for a in alphas:
        def hook(layer, x, _a=a):
            if layer != ell:
                return x
            cur = x[:, -1]
            out = x.clone()
            out[:, -1] = cur + _a * cur.norm(dim=1, keepdim=True) * d
            return out

        roll = model.rollout_with_edit_hook(b.state, hook, K_ROLL) if hasattr(
            model, "rollout_with_edit_hook") else _roll_hook(model, b.state, hook)
        recs.append({"editor": "Nanda addition", "point": ell, "alpha": a,
                     "write_ratio": float(a), **score(model, b, roll)})
    return recs


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
def pinv_single_point(model, b: Bench, probes: dict, alphas) -> list[dict]:
    """2. Pseudoinverse injection at ONE residual point, tried at every point, alpha swept.

    `Delta = A+(target - (A x + b))`, the minimum-norm null-space-preserving write, scaled by
    alpha. alpha = 1 is the exact jump that lands the read-out on the target; 2026-08-21 found the
    single-point optimum on Othello at alpha 1.5 with a ~50x spread, so the axis is swept.
    """
    recs = []
    for ell, (probe, _) in probes.items():
        W, bb, Wp = _decompose(probe)
        as_activations(model, ell)
        h0 = model.flat_state(b.state)
        err0 = float(((h0 @ W + bb) - b.tgt).norm(dim=1).mean())
        for a in alphas:
            # `inject_state` is the alpha=1 jump; alpha scales it as a lerp from h0. Calling the
            # canonical editor rather than re-deriving the formula is the point — the two were
            # verified equal to 1e-6 on this bench before the swap (2026-08-31).
            delta = a * (inject_state(h0, b.tgt, W.T, Wp.T, bb) - h0)
            h = h0 + delta
            roll = model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()
            recs.append({"editor": "PI injection (1 point)", "point": ell, "alpha": a,
                         "write_ratio": float((delta.norm(dim=1) / h0.norm(dim=1)).mean()),
                         "readout_err_before": err0,
                         "readout_err_after": float(((h @ W + bb) - b.tgt).norm(dim=1).mean()),
                         **score(model, b, roll)})
    return recs


def _decompose(probe):
    """(W, b, W+) in RAW activation space, from a linear `WorldStateProbe`."""
    lin = probe.net
    W = (lin.weight.detach().T / probe.x_std[:, None])
    bb = lin.bias.detach() - (probe.x_mean / probe.x_std) @ lin.weight.detach().T
    return W, bb, torch.linalg.pinv(W)


def grad_steering(model, b: Bench, probes: dict, start_layers, alphas, n_steps: int = 100,
                  beta: float = 0.2) -> list[dict]:
    """3. Li et al.'s MLP gradient steering at `L_s` and EVERY point after it.

    `othello_probe.build_edit_spec` + `make_intervention_hook`, unmodified — the same objects that
    reproduced their intervention on their own checkpoint.
    """
    recs = []
    for ls in start_layers:
        pts = {e: probes[e][0] for e in probes if e >= ls}
        for a in alphas:
            specs = {}
            for e, pr in pts.items():
                as_activations(model, e)
                specs[e] = op.build_edit_spec(pr, model.flat_state(b.state), b.change_mask,
                                              b.tgt, beta=beta)
            rec: dict = {}
            hook = op.make_intervention_hook(pts, specs, ls, alpha=a, n_steps=n_steps, record=rec)
            roll = _roll_hook(model, b.state, hook)
            recs.append({"editor": f"MLP grad steering @L_s={ls}", "point": ls, "alpha": a,
                         "write_ratio": float(np.mean(
                             [d["delta_norm"] / d["x_norm"] for d in rec.values()
                              if isinstance(d, dict) and d.get("x_norm", 0) > 0] or [np.nan])),
                         **score(model, b, roll)})
    return recs
