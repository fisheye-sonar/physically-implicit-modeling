"""pim.eval — model-agnostic evaluation: pure functions over numpy arrays.

Layout:
  _helpers.py         — inference (the only module that calls models)
  baselines.py        — reference RMSE baselines from dataset arrays
  prediction.py       — single-step / horizon / mse-by-context metrics
  recovery.py         — fit_probes, eval_recovery_multi
  rollout.py          — observation drift, position drift, coherence
  controllability.py  — warm_up_to_edit, rollout_steered/unsteered, eval_controllability
"""

from pim.eval._helpers import (
    autoregressive_rollout,
    autoregressive_rollouts,
    collect_rollouts,
    decode_states_multi,
    teacher_force,
)
from pim.eval.baselines import (
    ObsBaselines,
    PosBaselines,
    compute_obs_baselines,
    compute_pos_baselines,
)
from pim.eval.controllability import (
    ControllabilityMetrics,
    RolloutResult,
    WarmUpResult,
    eval_controllability,
    eval_position_controllability,
    rollout_gradient_steered,
    rollout_steered,
    rollout_unsteered,
    warm_up_to_edit,
)
from pim.eval.prediction import (
    PredictionMetrics,
    eval_horizon_mse,
    eval_mse_by_context,
    eval_single_step,
)
from pim.eval.recovery import (
    RecoveryMetrics,
    eval_recovery,
    eval_recovery_multi,
    fit_probes,
)
from pim.eval.rollout import (
    CoherenceMetrics,
    eval_observation_drift,
    eval_position_drift,
    eval_trajectory_coherence,
    per_sample_coherence,
    rollout_coherence,
)

__all__ = [
    # _helpers
    "teacher_force",
    "autoregressive_rollout",
    "autoregressive_rollouts",
    "collect_rollouts",
    "decode_states_multi",
    # baselines
    "ObsBaselines",
    "PosBaselines",
    "compute_obs_baselines",
    "compute_pos_baselines",
    # prediction
    "PredictionMetrics",
    "eval_single_step",
    "eval_horizon_mse",
    "eval_mse_by_context",
    # recovery
    "RecoveryMetrics",
    "fit_probes",
    "eval_recovery",
    "eval_recovery_multi",
    # rollout
    "CoherenceMetrics",
    "rollout_coherence",
    "per_sample_coherence",
    "eval_observation_drift",
    "eval_position_drift",
    "eval_trajectory_coherence",
    # controllability
    "WarmUpResult",
    "RolloutResult",
    "ControllabilityMetrics",
    "warm_up_to_edit",
    "rollout_steered",
    "rollout_unsteered",
    "rollout_gradient_steered",
    "eval_controllability",
    "eval_position_controllability",
]
