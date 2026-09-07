"""The two oracle editors stay usable — and stay ORACLES (2026-09-07).

They exist to defend the Edit Index: a write that provably carries the edited world must
score high, or a workhorse editor at the unedited floor would say nothing about the model.
Measured on the canonical noiseless run at n=192: overwrite +0.907 / freeze[N=16] +0.746
(dw-pn04: +0.688 / +0.678, against an effective ceiling of ~+0.82 because a noisy-trained
model is scored against the clean render). Pinned at >= +0.7 on the noiseless run.

Skipped when the run is not on disk (runs/ is not in git).
"""

from __future__ import annotations

from pathlib import Path

import pytest

RUN = Path("runs/noise_ablation/L-dw-noiseless-20m")
EVAL = Path("datasets/discworld/dw-noiseless/eval")
THRESHOLD = 0.7


@pytest.mark.skipif(not (RUN / "best_model.pt").exists() or not (EVAL / "edits.h5").exists(),
                    reason="canonical noiseless run / instance not on disk")
def test_both_oracle_editors_carry_the_edited_world():
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld import bench as dwb
    from pim.models import load_run

    model, _ = load_run(RUN, device=dwa.DEV)
    b = dwb.load_bench(model, n=192, target="full", basis_name="cartesian", data_dir=EVAL)
    u = dwa.unsteered(model, b)
    assert u["edit_index"] < -0.8                       # the floor is where it should be
    recs = {r["editor"]: r for r in dwa.oracle_arm(model, b, n_freeze=16)}
    for name, r in recs.items():
        assert r["edit_index"] >= THRESHOLD, f"{name}: EI {r['edit_index']:+.3f} < {THRESHOLD}"
        assert r["fidelity_ratio"] < 1.0, f"{name}: fidelity {r['fidelity_ratio']:.3f} (degraded)"
