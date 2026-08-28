"""Editability of a model on DISCWORLD — the counterpart to `envctrl_eval.py`.

Thin runner over `editability.py`, which already holds every editor and metric. Takes a run name
so any checkpoint under `runs/othello_arch/` can be scored on dataset 4's edits split — the same
edit set every other discworld number in this repo uses.

All three editors on **both** probe target sets (`pos` and `full`), so the target is an explicit
axis. That mirrors the fix made to `envctrl_eval.py` on 2026-08-22, where the Othello row had
silently used a different probe target for gradient steering than for the other two editors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import editability as E  # noqa: E402

from model import build as build_arch  # noqa: E402

A_ADD = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
A_PIN = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0)
# ^ wide enough that EVERY basis reaches |dh|/|h| ~ 1. Raw alpha is NOT comparable across
#   bases: the pinv write scales with ||target - A h||, and a teleport is ~2 units in x but
#   ~0.05 in 1/rho, so matched alpha means a 35x mismatched physical write.
A_GRD = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)
RUNS = _REPO / "runs" / "othello_arch"


def run(run_name: str, n_cases: int = 192, n_probe: int = 1500, log=print,
        basis_name: str = "cartesian", strict_probes: bool = True,
        probe_data_dir: str | None = None, linear_only: bool = False) -> dict:
    """`run_name` may be a bare name under runs/othello_arch/ or a repo-relative run dir.

    `linear_only` skips the MLP probe, and therefore also gradient steering — the only editor
    that consumes it. Nanda addition and PI injection both read the LINEAR probe and are
    unaffected. Roughly 6 min instead of 14 at n_probe=30,000. ⚠ It also disables the
    `MLP >= linear` tripwire, which needs both probes; use it for scouting, not for a number
    that has to stand on its own.

    `probe_data_dir` supplies a LARGER corpus for probe fitting only — the edit benchmark stays
    `datasets/4_fixed_refl_inview` regardless, so every Edit Index remains comparable to every
    other discworld number. dset 4's test split holds only 10,000 sequences, which is far short
    of the >=100k needed to keep a 262k-parameter MLP probe from memorising (2026-08-22: 1,500
    sequences gave in-sample vy R^2 0.954 against held-out -0.073).
    """
    t0 = time.time()
    rd = Path(run_name)
    if not rd.is_absolute():
        rd = (_REPO / run_name) if (_REPO / run_name).exists() else (RUNS / run_name)
    ck = torch.load(rd / "best_model.pt", map_location=E.DEV, weights_only=False)
    mc = ck.get("model_config") or {"obs_res": ck.get("obs_res"), "block_size": ck.get("block_size")}
    m = build_arch(obs_res=mc["obs_res"], block_size=mc["block_size"],
                   n_layer=mc.get("n_layer", 8), n_head=mc.get("n_head", 8),
                   n_embd=mc.get("n_embd", 512), dropout=mc.get("dropout", 0.1)).to(E.DEV)
    m.load_state_dict(ck["model_state"])
    m.eval()
    NP = E.n_points(m)
    log(f"{run_name}: {NP} residual points, val {ck['val_loss']:.5f}")
    out = {"run": run_name, "val_loss": float(ck["val_loss"]),
           "basis": basis_name, "targets": {}}
    sanity: list[dict] = []

    for target in ("pos", "full"):
        b = E.load_bench(m, n=n_cases, target=target, basis_name=basis_name)
        lin = E.fit_probes(m, target=target, n_seq=n_probe, hidden=None, log=None,
                           basis_name=basis_name, data_dir=probe_data_dir)
        mlp = None
        if not linear_only:
            mlp = E.fit_probes(m, target=target, n_seq=n_probe, hidden=512, log=None,
                               basis_name=basis_name, data_dir=probe_data_dir)
            # Tripwire, non-strict so the JSON still lands; `run()` raises after writing it.
            sanity.append(E.check_probe_sanity(
                lin, mlp, strict=False, log=log,
                label=f"{run_name} target={target} basis={basis_name} n_probe={n_probe:,}"))
        u = E.unsteered(m, b)
        log(f"  target={target}  UNEDITED EI {u['edit_index']:+.4f}  fidelity 1.000  "
            f"target {u['target_rmse']:.4f}  collat {u['collateral_rmse']:.4f}")
        recs = []
        for ell in range(NP):
            recs += E.nanda_addition(m, b, lin[ell][0], ell, A_ADD, b.out_dims)
        recs += E.pinv_single_point(m, b, lin, A_PIN)
        if not linear_only:
            recs += E.grad_steering(m, b, mlp, list(range(NP)), A_GRD, n_steps=100)
        for r in recs:
            r["fidelity_ratio"] = E.fidelity_ratio(r, u)
        out["targets"][target] = {
            "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
            "probe_r2_linear": [v[1]["r2"] for v in lin.values()],
            "probe_r2_mlp": ([v[1]["r2"] for v in mlp.values()] if mlp else None),
            # per-dim, so position and velocity can be read separately: dims are
            # [pos...] then [vel...] for target="full", pos only for target="pos"
            "probe_perdim_linear": [v[1]["per_dim_r2"] for v in lin.values()],
            "probe_perdim_mlp": ([v[1]["per_dim_r2"] for v in mlp.values()] if mlp else None),
            "probe_gap_mlp": ([v[1]["r2_insample"] - v[1]["r2"] for v in mlp.values()]
                              if mlp else None),
            "probe_gap_linear": [v[1]["r2_insample"] - v[1]["r2"] for v in lin.values()],
            "n_probe": n_probe,
            "arms": [{k: v for k, v in r.items() if np.isscalar(v)} for r in recs],
        }
        log(f"  {'editor':<26}{'pt':>4}{'alpha':>7}{'EI':>9}{'fid':>7}{'target':>9}{'collat':>9}")
        for ed in sorted({r["editor"].split(" @")[0] for r in recs}):
            sub = [r for r in recs if r["editor"].startswith(ed)]
            bst = max(sub, key=lambda r: r["edit_index"])
            log(f"  {ed:<26}{bst['point']:>4}{bst['alpha']:>7}{bst['edit_index']:>+9.4f}"
                f"{bst['fidelity_ratio']:>7.3f}{bst['target_rmse']:>9.4f}"
                f"{bst['collateral_rmse']:>9.4f}")
    out["probe_sanity"] = sanity
    (RUNS / f"{Path(run_name).name}_editability_{basis_name}_p{n_probe}"
            f"{'_lin' if linear_only else ''}.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"  wrote runs/othello_arch/{run_name}_editability.json [{(time.time() - t0) / 60:.1f} min]")
    # Raise AFTER the JSON is on disk: the diagnostic is worth keeping even when the run is
    # rejected, and a silent bad probe is what cost us the 2026-08-22 decodability numbers.
    nbad = sum(s["n_violations"] for s in sanity)
    if nbad and strict_probes:
        raise E.ProbeSanityError(
            f"{run_name}: {nbad} MLP-below-linear violation(s) across targets — decodability "
            f"numbers in {run_name}_editability_{basis_name}_p{n_probe}.json are NOT trustworthy. "
            f"Refit with more probe sequences (n_probe >= 10000).")
    return out


if __name__ == "__main__":
    # usage: discworld_eval.py <run> [bases,csv] [n_probe,csv] [probe_data_dir] [strict]
    args = sys.argv[1:] or ["L90_theirs_discworld"]
    name = args[0]
    bases = args[1].split(",") if len(args) > 1 else ["cartesian"]
    probes = [int(x) for x in args[2].split(",")] if len(args) > 2 else [1500]
    pdd = args[3] if len(args) > 3 and args[3] != "-" else None
    strict = (len(args) > 4 and args[4] == "strict")
    lin_only = (len(args) > 5 and args[5] == "linear")
    for np_ in probes:
        for bname in bases:
            print(f"\n{'=' * 74}\nBASIS: {bname}   PROBE SEQUENCES: {np_:,}\n{'=' * 74}",
                  flush=True)
            run(name, basis_name=bname, n_probe=np_, probe_data_dir=pdd,
                strict_probes=strict, linear_only=lin_only)
