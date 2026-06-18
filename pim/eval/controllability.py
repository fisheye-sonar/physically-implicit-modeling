"""Counterfactual controllability evaluation.

Can we causally steer the model's internal state via probe injection, and
does the model's subsequent rollout track the post-edit ground truth?

Pipeline (called in sequence from the notebook):

  warm_up_to_edit   — teacher-force each edit sample to edit_frame, collect h
  rollout_steered   — inject target via probe pseudoinverse, then roll out
  rollout_unsteered — roll out from the un-edited state (control)
  eval_controllability         — scalar + per-step observation RMSE summaries
  eval_position_controllability — per-step position RMSE for each probe

All rollouts are model-agnostic via the HiddenStateModel SSM protocol
(state_from_flat / decode / predict_step). Step-0 = decode(state) without
advancing, so the linear probe at step 0 reads the injected target exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from pim.editors.gradient_steering import gradient_steer
from pim.editors.probe_steering import inject_state, probe_decomposition
from pim.extractors.linear import LinearExtractor
from pim.world_models.protocol import HiddenStateModel


@dataclass
class WarmUpResult:
    """Hidden states from teacher-forcing each edit sample to edit_frame."""
    h_at_edit: np.ndarray            # (N, H)
    h_pre_edit: np.ndarray           # (n_viz, n_ctx_show, H) — last frames before edit
    n_ctx_show: int


@dataclass
class RolloutResult:
    """Per-step observations and hidden states from a rollout starting at edit_frame.

    Step 0 = decode(state) without advancing.
    """
    obs: np.ndarray                  # (N, n_rollout, R)
    h: np.ndarray                    # (N, n_rollout, H)
    injection_error: float = 0.0     # mean squared probe(h_edited) - target; 0 for unsteered


@dataclass
class ControllabilityMetrics:
    """Scalar + per-step observation RMSE summary."""
    steered_mse: float
    unsteered_mse: float
    injection_error: float
    steered_obs_step: np.ndarray         # (n_rollout,) RMSE per step vs noisy gt
    unsteered_obs_step: np.ndarray
    clean_steered_obs_step: np.ndarray   # vs clean gt
    clean_unsteered_obs_step: np.ndarray


@torch.no_grad()
def warm_up_to_edit(
    model: HiddenStateModel,
    obs_seqs: np.ndarray,           # (N, T, R)
    edit_frame: int,
    *,
    n_viz: int = 3,
    n_ctx_show: int = 8,
    device: str = "cpu",
    desc: str = "warm-up to edit",
) -> WarmUpResult:
    """Teacher-force each sample to edit_frame; collect flat hidden state at edit.

    Also captures the last n_ctx_show pre-edit hidden states for the first
    n_viz samples (used by trajectory viz plots).
    """
    N = obs_seqs.shape[0]
    H = model.hidden_size
    n_ctx_show = min(n_ctx_show, edit_frame)
    n_viz = min(n_viz, N)

    h_at_edit = np.zeros((N, H), dtype=np.float32)
    h_pre_edit = np.zeros((n_viz, n_ctx_show, H), dtype=np.float32)

    for i in tqdm(range(N), desc=desc, leave=False):
        obs_t = torch.from_numpy(obs_seqs[i]).float().to(device)
        state = None
        for t in range(edit_frame):
            _, state = model.step(obs_t[t].unsqueeze(0), state)
            if i < n_viz and t >= edit_frame - n_ctx_show:
                h_pre_edit[i, t - (edit_frame - n_ctx_show)] = (
                    model.flat_state(state).squeeze(0).cpu().numpy()
                )
        h_at_edit[i] = model.flat_state(state).squeeze(0).cpu().numpy()

    return WarmUpResult(h_at_edit=h_at_edit, h_pre_edit=h_pre_edit, n_ctx_show=n_ctx_show)


@torch.no_grad()
def _rollout(
    model: HiddenStateModel,
    h_flat: torch.Tensor,           # (1, H)
    n_rollout: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Step-0 = decode(state); subsequent steps = predict_step."""
    state = model.state_from_flat(h_flat)
    obs = [model.decode(state).squeeze(0).cpu().numpy()]
    h = [model.flat_state(state).squeeze(0).cpu().numpy()]
    for _ in range(n_rollout - 1):
        pred, state = model.predict_step(state)
        obs.append(pred.squeeze(0).cpu().numpy())
        h.append(model.flat_state(state).squeeze(0).cpu().numpy())
    return np.stack(obs), np.stack(h)


@torch.no_grad()
def rollout_steered(
    model: HiddenStateModel,
    h_at_edit: np.ndarray,           # (N, H)
    targets: np.ndarray,             # (N, output_dim)
    linear_probe: LinearExtractor,
    n_rollout: int,
    *,
    device: str = "cpu",
    desc: str = "steered rollouts",
) -> RolloutResult:
    """Inject target into each h via probe pseudoinverse, then roll out."""
    linear_probe = linear_probe.to(device).eval()
    A, b, A_pinv = probe_decomposition(linear_probe)

    N = h_at_edit.shape[0]
    all_obs, all_h, inj_errs = [], [], []
    for i in tqdm(range(N), desc=desc, leave=False):
        h = torch.from_numpy(h_at_edit[i]).float().to(device).unsqueeze(0)
        tgt = torch.from_numpy(targets[i]).float().to(device).unsqueeze(0)
        h_edited = inject_state(h, tgt, A, A_pinv, b)
        readback = (h_edited @ A.T + b).squeeze(0)
        inj_errs.append(float(((readback - tgt.squeeze(0)) ** 2).mean().item()))
        obs, hs = _rollout(model, h_edited, n_rollout)
        all_obs.append(obs)
        all_h.append(hs)
    return RolloutResult(
        obs=np.stack(all_obs),
        h=np.stack(all_h),
        injection_error=float(np.mean(inj_errs)),
    )


def rollout_gradient_steered(
    model: HiddenStateModel,
    h_at_edit: np.ndarray,           # (N, H)
    targets: np.ndarray,             # (N, output_dim)
    probe: torch.nn.Module,
    n_rollout: int,
    *,
    n_steps: int = 200,
    lr: float = 0.01,
    reg_weight: float = 0.0,
    device: str = "cpu",
    desc: str = "gradient-steered rollouts",
) -> RolloutResult:
    """Gradient-steer each h to match target via the probe, then roll out.

    Optimises h* = argmin ||probe(h*) - target||² + reg_weight*||h* - h_orig||²
    using Adam, then runs an unguided rollout from h*.

    Parameters
    ----------
    probe       : any differentiable extractor (MLPExtractor or LinearExtractor).
    n_steps     : gradient steps per sample.
    lr          : Adam learning rate.
    reg_weight  : L2 anchor weight (0 = disabled).
    """
    probe = probe.to(device).eval()
    N = h_at_edit.shape[0]
    all_obs, all_h, inj_errs = [], [], []
    for i in tqdm(range(N), desc=desc, leave=False):
        h = torch.from_numpy(h_at_edit[i]).float().to(device).unsqueeze(0)
        tgt = torch.from_numpy(targets[i]).float().to(device).unsqueeze(0).reshape(1, -1)
        h_edited, final_mse = gradient_steer(
            h, tgt, probe,
            n_steps=n_steps, lr=lr, reg_weight=reg_weight,
        )
        inj_errs.append(final_mse)
        with torch.no_grad():
            obs, hs = _rollout(model, h_edited, n_rollout)
        all_obs.append(obs)
        all_h.append(hs)
    return RolloutResult(
        obs=np.stack(all_obs),
        h=np.stack(all_h),
        injection_error=float(np.mean(inj_errs)),
    )


@torch.no_grad()
def rollout_unsteered(
    model: HiddenStateModel,
    h_at_edit: np.ndarray,           # (N, H)
    n_rollout: int,
    *,
    device: str = "cpu",
    desc: str = "unsteered rollouts",
) -> RolloutResult:
    """Roll out from the un-injected state (control)."""
    N = h_at_edit.shape[0]
    all_obs, all_h = [], []
    for i in tqdm(range(N), desc=desc, leave=False):
        h = torch.from_numpy(h_at_edit[i]).float().to(device).unsqueeze(0)
        obs, hs = _rollout(model, h, n_rollout)
        all_obs.append(obs)
        all_h.append(hs)
    return RolloutResult(obs=np.stack(all_obs), h=np.stack(all_h))


def eval_controllability(
    steered: RolloutResult,
    unsteered: RolloutResult,
    gt_obs: np.ndarray,              # (N, n_rollout, R) noisy ground truth from edit_frame on
    clean_gt_obs: np.ndarray,        # (N, n_rollout, R) clean ground truth
) -> ControllabilityMetrics:
    """Per-step + scalar RMSE summaries comparing steered/unsteered to GT."""
    steered_obs_step = ((steered.obs - gt_obs) ** 2).mean(axis=(0, 2))
    unsteered_obs_step = ((unsteered.obs - gt_obs) ** 2).mean(axis=(0, 2))
    clean_steered_obs_step = ((steered.obs - clean_gt_obs) ** 2).mean(axis=(0, 2))
    clean_unsteered_obs_step = ((unsteered.obs - clean_gt_obs) ** 2).mean(axis=(0, 2))
    return ControllabilityMetrics(
        steered_mse=float(((steered.obs - gt_obs) ** 2).mean()),
        unsteered_mse=float(((unsteered.obs - gt_obs) ** 2).mean()),
        injection_error=steered.injection_error,
        steered_obs_step=steered_obs_step,
        unsteered_obs_step=unsteered_obs_step,
        clean_steered_obs_step=clean_steered_obs_step,
        clean_unsteered_obs_step=clean_unsteered_obs_step,
    )


def eval_position_controllability(
    steered: RolloutResult,
    unsteered: RolloutResult,
    gt_positions: np.ndarray,        # (N, n_rollout, n_obj, 2)
    probes,                          # list[ProbeSpec]
    device: str = "cpu",
) -> dict[str, dict[str, np.ndarray]]:
    """Per-step position RMSE for each probe, for steered + unsteered rollouts.

    Returns {probe.name: {"steered": (n_rollout,), "unsteered": (n_rollout,)}}.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        h_s = torch.from_numpy(steered.h).float().to(device)
        h_u = torch.from_numpy(unsteered.h).float().to(device)
        for p in probes:
            p.probe.to(device).eval()
            pos_s = p.probe(h_s).cpu().numpy()
            pos_u = p.probe(h_u).cpu().numpy()
            out[p.name] = {
                "steered":   ((pos_s - gt_positions) ** 2).mean(axis=(0, 2, 3)),
                "unsteered": ((pos_u - gt_positions) ** 2).mean(axis=(0, 2, 3)),
            }
    return out
