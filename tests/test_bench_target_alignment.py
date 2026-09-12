"""The discworld write target is the PRE-dynamics state (2026-09-12).

The probe at the edit point reads the state that rendered the last consumed frame; the
model's next output is one dynamics step ahead. So the target the editors write must be the
state that, advanced one step by the simulator, renders the stored edited frame — for the
edited object pos[EF] − v·dt, for the other object its current position. These tests pin
that on the real noiseless instance (exact dynamics) and on a synthetic bench-free check.
Skipped where the instance is absent (datasets/ is not in git).
"""
from __future__ import annotations

import numpy as np
import pytest

from pim.environments import layout
from pim.environments.discworld import bench as dwb

INST = "dw-noiseless"
pytestmark = pytest.mark.skipif(not layout.edits_file("discworld", INST).exists(),
                                reason="dw-noiseless edit bench not on disk")


def _pre_dyn(a):
    ar = np.arange(a["n"])
    k = a["edit_object"]
    pre = a["pos"][:, dwb.EF - 1].copy()
    pre[ar, k] = a["pos"][ar, dwb.EF, k] - a["vel"][ar, dwb.EF - 1, k] * float(a["sim"]["dt"])
    return pre


def test_regression_target_is_the_pre_dynamics_state():
    a = dwb.bench_arrays(n=48, target="pos", basis_name="cartesian", instance=INST)
    pre = _pre_dyn(a)
    assert np.allclose(a["y"], pre.reshape(a["n"], -1), atol=1e-6)
    # the OTHER object is asked to HOLD (its current position), not to advance
    ar, k = np.arange(a["n"]), a["edit_object"]
    other = 1 - k
    assert np.allclose(a["y"].reshape(a["n"], 2, 2)[ar, other], a["pos"][ar, dwb.EF - 1, other])
    # the edited object's target is exactly one step short of where it appears
    assert np.allclose(a["y"].reshape(a["n"], 2, 2)[ar, k] + a["vel"][ar, dwb.EF - 1, k] * a["sim"]["dt"],
                       a["pos"][ar, dwb.EF, k], atol=1e-6)


def test_target_advanced_one_step_renders_the_stored_edited_frame():
    """Noiseless instance: simulate one step from the target state and render — it must equal
    the bench's edited reference (the model is compared with exactly this frame)."""
    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import object_constants, sim_config_from

    a = dwb.bench_arrays(n=64, target="full", basis_name="frustum", instance=INST)
    pre = _pre_dyn(a)
    nxt = pre + a["vel"][:, dwb.EF - 1] * float(a["sim"]["dt"])          # one exact dynamics step
    cfg = sim_config_from(a["sim"], 2)
    rad, refl = object_constants(a["sim"], 2)
    for i in range(a["n"]):
        r = render_frame(nxt[i].astype(np.float32), rad, refl, cfg)[2]
        assert np.allclose(r, a["zones"].gt_edited[i], atol=1e-5), f"case {i}"
    # and the frustum-basis target is the basis image of that same pre-dynamics state
    bp, bv = dwb._to_basis(pre, a["vel"][:, dwb.EF - 1], a["sim"], "frustum")
    assert np.allclose(a["y"][:, :4], bp.reshape(a["n"], -1), atol=1e-5)
    assert np.allclose(a["y"][:, 4:], bv.reshape(a["n"], -1), atol=1e-5)


def test_categorical_target_cell_is_the_pre_dynamics_cell():
    from pim.environments.discworld.grid_target import categorical_target

    grid = categorical_target("grid-16x8")
    a = dwb.bench_arrays(n=32, target="grid-16x8", basis_name="frustum", instance=INST)
    pre = _pre_dyn(a)
    ar, k = np.arange(a["n"]), a["edit_object"]
    assert np.array_equal(a["cells"]["B"], grid.cell_of(pre[ar, k], a["sim"]))
    assert np.array_equal(a["cells"]["A"], grid.cell_of(a["pos"][ar, dwb.EF - 1, k], a["sim"]))
    assert (a["cells"]["A"] != a["cells"]["B"]).all()          # the selection rule, on the new target
