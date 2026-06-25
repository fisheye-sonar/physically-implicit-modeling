"""pim.editors — methods for steering / modifying model internal states."""

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
from pim.editors.probe_steering import probe_decomposition, decompose_hidden, inject_state

__all__ = [
    "gradient_steer",
    "probe_decomposition",
    "decompose_hidden",
    "inject_state",
    "StateSubspace",
    "fit_state_subspace",
    "fit_local_subspace",
    "manifold_steer",
    "manifold_steer_local",
    "offmanifold_residual",
    "project_to_subspace",
]
