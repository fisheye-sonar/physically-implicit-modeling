"""pim.editors — the canonical editors: how a write to model internals is attempted.

The three workhorses every analysis defaults to (one file each):

    pinv.py         PI — pseudoinverse injection: the minimum-norm write that lands
                    the linear read-out (z-space + y-affine canonical; the named
                    "legacy" space reproduces pre-2026-08-31 discworld numbers).
    nanda.py        ND — Nanda et al. direction addition: one vector along the probe
                    weight row, no solve, no gradients.
    grad_steer.py   GS — Li et al. MLP gradient steering, sequential across layers.

Three kept non-default editors:

    nullspace.py            multi-probe write to the WHOLE linear code (pairs with
                            pim.probes.nullspace) — answers the row-space objection.
    oracle_overwrite.py     counterfactual state overwrite (oracle ceiling).
    freeze_interpolation.py freeze-time teacher-forced interpolation (oracle,
                            through the observation channel).

Legacy modules (`probe_steering`, `gradient_steering`, `manifold_steering`) remain
importable during the 2026-08/09 housecleaning transition and are removed with it.
"""

# ── canonical ────────────────────────────────────────────────────────────────
from pim.editors.freeze_interpolation import freeze_time_rollout, frozen_frames
from pim.editors.grad_steer import (
    EditSpec,
    build_edit_spec,
    make_intervention_hook,
    rollout_with_sequential_intervention,
)
from pim.editors.nanda import addition_delta, addition_hook, probe_direction
from pim.editors.nullspace import multiprobe_delta
from pim.editors.oracle_overwrite import counterfactual_state, overwrite_rollout
from pim.editors.pinv import (
    PinvMap,
    decompose_hidden,
    inject_state,
    pinv_maps,
    pinv_step,
    readout_error,
)

# ── legacy (transition only) ─────────────────────────────────────────────────
from pim.editors.gradient_steering import gradient_steer
from pim.editors.manifold_steering import (
    StateSubspace,
    fit_local_subspace,
    fit_state_subspace,
    manifold_steer,
    manifold_steer_local,
    offmanifold_residual,
    project_to_subspace,
)
from pim.editors.probe_steering import probe_decomposition

__all__ = [
    # PI
    "PinvMap",
    "pinv_maps",
    "pinv_step",
    "inject_state",
    "decompose_hidden",
    "readout_error",
    # ND
    "probe_direction",
    "addition_delta",
    "addition_hook",
    # GS
    "EditSpec",
    "build_edit_spec",
    "make_intervention_hook",
    "rollout_with_sequential_intervention",
    # non-default
    "multiprobe_delta",
    "counterfactual_state",
    "overwrite_rollout",
    "frozen_frames",
    "freeze_time_rollout",
    # legacy
    "gradient_steer",
    "probe_decomposition",
    "StateSubspace",
    "fit_state_subspace",
    "fit_local_subspace",
    "manifold_steer",
    "manifold_steer_local",
    "offmanifold_residual",
    "project_to_subspace",
]
