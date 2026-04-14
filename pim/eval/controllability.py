"""Counterfactual controllability evaluation.

Criterion 4: can we causally steer the model's internal state by injecting
a target env state via the probe, and does the model's subsequent rollout
track the post-edit ground truth?

The edit operation (inject_state) is in pim/editors/probe_steering.py.
This module handles the rollout and metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pim.editors.probe_steering import inject_state, probe_decomposition
from pim.extractors.linear import LinearExtractor
from pim.world_models.protocol import WorldModel


@dataclass
class ControllabilityMetrics:
    """Results from a controllability evaluation."""
    steered_mse: float          # AR rollout MSE after probe-steered edit
    unsteered_mse: float        # AR rollout MSE with no edit (baseline)
    injection_error: float      # how accurately the probe reads back the injected state


def eval_controllability(
    internal_states_at_edit: np.ndarray,  # (N, H) hidden state just before edit frame
    env_state_targets: np.ndarray,        # (N, output_dim) desired state (flattened)
    extractor: LinearExtractor,
    model: WorldModel,
    obs_post_edit_actual: np.ndarray,     # (N, T_rollout, R) ground truth after edit
    obs_at_edit: np.ndarray,              # (N, R) observation at edit frame
    *,
    n_rollout: int = 15,
    device: str = "cpu",
) -> ControllabilityMetrics:
    """Evaluate hidden-state steering via probe decomposition.

    For each sample:
    1. Inject the target state into the hidden state at the edit frame.
    2. Roll out the model autoregressively for n_rollout steps from the edited state.
    3. Compare steered rollout vs ground truth post-edit observations.
    4. Also roll out without editing (unsteered baseline).

    Parameters
    ----------
    internal_states_at_edit : (N, H) hidden states just before the edit frame
    env_state_targets       : (N, output_dim) target env state (flattened state_shape)
    extractor               : trained LinearExtractor (required for pseudoinverse)
    model                   : WorldModel for post-edit rollout
    obs_post_edit_actual    : (N, T_rollout, R) ground truth post-edit observations
    obs_at_edit             : (N, R) observation at the edit frame (used to start
                              unsteered rollout from the same point)
    n_rollout               : steps to roll out after edit
    device                  : torch device string

    Returns
    -------
    ControllabilityMetrics
    """
    extractor = extractor.to(device).eval()
    A, b, A_pinv = probe_decomposition(extractor)

    N = internal_states_at_edit.shape[0]
    steered_errs, unsteered_errs, inj_errs = [], [], []

    with torch.no_grad():
        for i in range(N):
            h = torch.from_numpy(internal_states_at_edit[i]).float().to(device)  # (H,)
            target = torch.from_numpy(env_state_targets[i]).float().to(device)   # (D,)

            # Injection error: does the probe read back the target?
            h_edited = inject_state(h.unsqueeze(0), target.unsqueeze(0), A, A_pinv, b)
            readback = (h_edited @ A.T + b).squeeze(0)
            inj_err = float(((readback - target) ** 2).mean().item())
            inj_errs.append(inj_err)

            # Steered rollout from edited hidden state
            # Reshape h_edited to GRU format (num_layers=1, batch=1, H)
            h_gru = h_edited.unsqueeze(0)   # (1, 1, H)
            obs_start = torch.from_numpy(obs_at_edit[i]).float().to(device).unsqueeze(0)  # (1, R)
            preds_steered = _rollout_from_h(model, h_gru, obs_start, n_rollout)

            # Unsteered rollout from original hidden state
            h_gru_orig = h.unsqueeze(0).unsqueeze(0)   # (1, 1, H)
            preds_unsteered = _rollout_from_h(model, h_gru_orig, obs_start, n_rollout)

            gt = obs_post_edit_actual[i, :n_rollout]   # (n_rollout, R)
            steered_errs.append(float(np.mean((preds_steered - gt) ** 2)))
            unsteered_errs.append(float(np.mean((preds_unsteered - gt) ** 2)))

    return ControllabilityMetrics(
        steered_mse=float(np.mean(steered_errs)),
        unsteered_mse=float(np.mean(unsteered_errs)),
        injection_error=float(np.mean(inj_errs)),
    )


@torch.no_grad()
def _rollout_from_h(
    model: WorldModel,
    h: torch.Tensor,      # (num_layers, 1, H)
    obs_start: torch.Tensor,  # (1, R)
    n_rollout: int,
) -> np.ndarray:
    """Roll out model for n_rollout steps starting from hidden state h."""
    preds = []
    x = obs_start
    for _ in range(n_rollout):
        x, h = model.step(x, h)
        preds.append(x.squeeze(0).cpu().numpy())
    return np.stack(preds, axis=0)   # (n_rollout, R)
