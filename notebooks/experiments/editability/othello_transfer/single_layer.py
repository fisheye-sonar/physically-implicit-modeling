"""Pseudoinverse injection applied at ONE residual point only, for each of the 9 points.

The main sweep re-imposes "hold all 64 tiles at their current read-out" at every layer, which may
actively undo the edit's own propagation (Nanda's fixed direction cannot do that). This isolates
it: write once, at one point, and let the remaining blocks propagate freely.

Nanda's direction addition is run at the same single points as a control — otherwise a single-layer
failure cannot be attributed to the pseudoinverse rather than to single-layer writes in general.
"""
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_HERE.parents[3])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_data as od  # noqa: E402
import transfer_pipeline as tp  # noqa: E402
from linear_intervention import case_targets, load_linear_probes, run  # noqa: E402

A_PINV = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
A_ADD = (0.05, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0)

shim = tp.load_model()
bench = od.load_benchmark()
probes = load_linear_probes()
cur_lab, tgt_lab = case_targets(bench)
uns = od.scorecard(tp.unsteered(shim, bench), bench)
print(f"\nnull: Li {uns['li_error_vs_post']:.3f}   Edit Index {uns['edit_index_union']:+.3f}")
print("reference, ALL 9 points: pinv best Li 1.461 / EI -0.275 · Nanda best Li 0.062 / EI +0.603\n")

t0 = time.time()
for mode, alphas, tag in (("pinv", A_PINV, "OURS pseudoinverse"), ("add", A_ADD, "Nanda addition")):
    print(f"=== {tag}, written at ONE residual point ===")
    print(f"{'point':>6} {'best a':>7} {'|dx|/|x|':>9} {'Li post':>8} {'Li pre':>8} "
          f"{'EI union':>9} {'EI symd':>8} {'legal':>6}")
    for ell in range(tp.N_POINTS):
        best, bc = None, None
        for a in alphas:
            _, c = run(shim, bench, probes, mode, a, {ell}, tgt_lab, cur_lab)
            if bc is None or c["li_error_vs_post"] < bc["li_error_vs_post"]:
                best, bc = a, c
        print(f"{ell:>6} {best:>7} {bc['write_ratio']:>9.3f} {bc['li_error_vs_post']:>8.3f} "
              f"{bc['li_error_vs_pre']:>8.3f} {bc['edit_index_union']:>+9.3f} "
              f"{bc['edit_index_symdiff']:>+8.3f} {bc['legal_mass']:>6.3f}", flush=True)
    print()
print(f"total {time.time()-t0:.0f}s")
