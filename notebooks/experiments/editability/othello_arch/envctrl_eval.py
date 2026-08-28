"""Editability of THEIR architecture on OTHELLO — all three editors on a COMMON probe target.

The environment control's evaluation, rewritten to remove a confound in the 2026-08-22 first pass.

⛔ **What was wrong the first time.** The three editors did not share a probe target:

    Nanda addition / PI injection  ->  ("mine",  "linear",         "frame", p)
    MLP gradient steering          ->  ("state", "MLP 512 hidden", "frame", p)

That is faithful to the two papers — Li et al. steer on absolute colour, Nanda et al. on
mine/theirs — but it makes the *within-row* editor comparison uninterpretable: gradient steering's
−0.010 could be a target artifact rather than a fact about the editor. The discworld row never had
this problem (all three editors shared `pos`/`full` there), so the apparent "editor ranking
inversion between environments" was partly self-inflicted.

Here every editor is run on **every** (target, family) combination it can use, so the target is an
explicit axis instead of a hidden one, and the gradient editor gets the same alpha budget as the
others rather than a quarter of it.

Everything else is imported: probes from `othello_gpt/othello_probe`, the gradient editor from the
same module, the pseudoinverse from `pim.editors.probe_steering`, the benchmark and every metric
from `othello_transfer/othello_data`. `OthelloGPTShim` puts the minGPT behind our names.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
# ⚠ ours_on_othello/model.py and othello_arch/model.py are BOTH named `model`; `sys.path.insert(0)`
#   means the LAST inserted wins, so ours_on_othello must go in last or `evaluate` gets the wrong
#   one and dies importing BLOCK.
for _p in (str(_HERE.parent / "othello_transfer"), str(_HERE), str(_REPO),
           str(_HERE.parent / "ours_on_othello")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluate as ev  # noqa: E402  (ours_on_othello)
import linear_intervention as li  # noqa: E402
import othello_data as od  # noqa: E402
import transfer_pipeline as tp  # noqa: E402
from othello_shim import OthelloGPTShim  # noqa: E402

from model_othello import build as build_theirs  # noqa: E402

A_ADD = (0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0)
A_PIN = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
A_GRD = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)   # same budget as the others
RUNS = _REPO / "runs" / "othello_arch"



def _fingerprint(model) -> str:
    """12 hex chars over every parameter — a model's identity, cheaply.

    Same construction as `ours_on_othello/evaluate.py:fingerprint` and
    `othello_arch/editability.py:_fingerprint`.
    """
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load(run: str):
    # `run` may be a bare name under runs/othello_arch/, or a repo-relative path to any run dir
    # (BIG20M_othello_L lives under runs/scaling/), so resolve both.
    rd = Path(run) if (Path(run).is_absolute() or (_REPO / run).exists()) else RUNS / run
    if not rd.is_absolute():
        rd = _REPO / rd
    ck = torch.load(rd / "best_model.pt", map_location=ev.DEV, weights_only=False)
    native = build_theirs().to(ev.DEV)
    native.load_state_dict(ck["model_state"])
    native.eval()
    shim = OthelloGPTShim(native.gpt).to(ev.DEV).eval()
    tp.N_POINTS = li.N_POINTS = shim.n_layers + 1
    # ⛔ PER-MODEL cache dir, keyed on a hash of the weights.
    # `fit_probe_grid`'s own key covers the probe settings and the data but NOT the model, and
    # this file used to point every checkpoint at ONE shared directory. Two checkpoints with the
    # same probe settings therefore hash to the same file, and the second silently gets the
    # first's probes. That is exactly the 2026-08-21 failure (a random-init control was served
    # the trained model's probes; both reported identical error), which was fixed in
    # `ours_on_othello/evaluate.py` but never here. Found 2026-08-24 while costing the sweep on
    # BIG20M_othello_L, which would have loaded L90_theirs_othello's probes.
    tp.CACHE_DIR = RUNS / "probe_cache" / _fingerprint(native)
    tp.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return shim, ck


def run(run_name: str, n_probe_games: int = 20_000, log=print,
        targets: tuple[str, ...] = ("state", "mine")) -> dict:
    """`targets` selects the probe target axis; `("mine",)` halves the grid to 18 probes.

    ⚠ `n_probe_games` defaults to 20,000, NOT the 6,000 used through 2026-08-22. At 6,000 the
    MLP-512 probe has 361,152 parameters against 353,900 rows (0.98 rows/param) and **loses to
    the linear probe on held-out data** — measured at point 6 on BIG20M_othello_L: MLP 1.617%
    held-out against 0.623% in-sample, linear 1.338%. It is fitting the probe training set.
    At 20,000 games (3.27 rows/param) the MLP recovers to 1.362% and ties linear.

    Gradient steering is the editor that consumes the MLP probe, so every gradient-steering
    number taken at 6,000 games — including the 2026-08-22 "fails on both targets" — was
    measured through an overfit probe and needs re-running.
    """
    t0 = time.time()
    shim, ck = load(run_name)
    NP = tp.N_POINTS
    log(f"{run_name}: {NP} residual points, val {ck['val_loss']:.5f}")
    data = ev.probe_data(n_probe_games)
    bench = od.load_benchmark()
    cur, tgt = ev.case_targets(bench)
    grid = tp.fit_probe_grid(shim, data, targets=targets,
                             families=("MLP 512 hidden", "linear"), splits=("frame",),
                             epochs=200, log=lambda s: None)
    uns = od.scorecard(tp.unsteered(shim, bench), bench)
    log(f"  UNEDITED  Li {uns['li_error_vs_post']:.3f}  EI {uns['edit_index_union']:+.4f}  "
        f"legal {uns['legal_mass']:.3f}")
    out = {"run": run_name, "val_loss": float(ck["val_loss"]),
           "unedited": {k: v for k, v in uns.items() if np.isscalar(v)},
           "n_probe_games": int(n_probe_games), "targets": list(targets),
           "probe": [{k: s[k] for k in ("target", "family", "point", "error_rate",
                                        "error_rate_insample")}
                     for s in grid.stats],
           "arms": []}
    allp = set(range(NP))

    # Direction addition and the pseudoinverse need a LINEAR probe; run them on BOTH targets.
    for target in targets:
        lin = {p: grid.probes[(target, "linear", "frame", p)] for p in range(NP)}
        for a in A_ADD:
            for sub, name in ((False, "Nanda addition"), (True, "Nanda (target-current)")):
                _, c = li.run(shim, bench, lin, "add", a, allp, tgt, cur, sub)
                out["arms"].append({"editor": name, "target": target, "alpha": a, "point": "all",
                                    **{k: v for k, v in c.items() if np.isscalar(v)}})
        for a in A_PIN:
            for p in range(NP):
                _, c = li.run(shim, bench, lin, "pinv", a, {p}, tgt, cur)
                out["arms"].append({"editor": "PI injection (1 point)", "target": target,
                                    "alpha": a, "point": p,
                                    **{k: v for k, v in c.items() if np.isscalar(v)}})
        log(f"  direction/PI on target={target} done [{(time.time() - t0) / 60:.1f} min]")

    # Gradient steering needs an MLP probe; same two targets, same alpha budget.
    for target in targets:
        mlp = {p: grid.probes[(target, "MLP 512 hidden", "frame", p)] for p in range(NP)}
        for ls in range(NP):
            for a in A_GRD:
                pr, _ = tp.run_arm(shim, bench, mlp, ls, alpha=a, n_steps=100, beta=0.2)
                c = od.scorecard(pr, bench)
                out["arms"].append({"editor": "MLP grad steering", "target": target,
                                    "alpha": a, "point": ls,
                                    **{k: v for k, v in c.items() if np.isscalar(v)}})
        log(f"  grad steering on target={target} done [{(time.time() - t0) / 60:.1f} min]")

    log(f"  {'editor':<26}{'target':>7}{'pt':>5}{'alpha':>7}{'EI':>9}{'Li':>8}"
        f"{'Li-pre':>8}{'legal':>7}")
    for ed in sorted({a["editor"] for a in out["arms"]}):
        for target in targets:
            sub = [a for a in out["arms"] if a["editor"] == ed and a["target"] == target]
            if not sub:
                continue
            b = max(sub, key=lambda r: r["edit_index_union"])
            log(f"  {ed:<26}{target:>7}{str(b['point']):>5}{b['alpha']:>7}"
                f"{b['edit_index_union']:>+9.4f}{b['li_error_vs_post']:>8.3f}"
                f"{b['li_error_vs_pre']:>8.3f}{b['legal_mass']:>7.3f}")
    stem = Path(run_name).name          # `run_name` may be a path; the JSON is named by the run
    tag = "" if len(targets) > 1 else f"_{targets[0]}"
    outp = RUNS / f"{stem}_editability{tag}.json"
    outp.write_text(json.dumps(out, indent=1, default=float))
    log(f"  wrote {outp.relative_to(_REPO)}  [{(time.time() - t0) / 60:.1f} min]")
    return out


if __name__ == "__main__":
    # usage: envctrl_eval.py <run-name-or-path> [targets, comma-separated] [n_probe_games]
    a = sys.argv[1:]
    name = a[0] if a else "ENVCTRL14_theirs_900k"
    tgts = tuple(a[1].split(",")) if len(a) > 1 else ("state", "mine")
    ngames = int(a[2]) if len(a) > 2 else 20_000
    run(name, n_probe_games=ngames, targets=tgts)
