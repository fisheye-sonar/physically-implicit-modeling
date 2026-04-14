"""pim.eval — model-agnostic evaluation modules."""

from pim.eval._helpers import run_autoregressive, run_teacher_forcing, collect_rollout
from pim.eval.prediction import eval_single_step, eval_horizon_mse, eval_mse_by_context, PredictionMetrics
from pim.eval.recovery import eval_recovery, RecoveryMetrics
from pim.eval.rollout import eval_observation_drift, eval_trajectory_coherence, rollout_coherence, CoherenceMetrics
from pim.eval.controllability import eval_controllability, ControllabilityMetrics

__all__ = [
    "run_autoregressive",
    "run_teacher_forcing",
    "collect_rollout",
    "eval_single_step",
    "eval_horizon_mse",
    "eval_mse_by_context",
    "PredictionMetrics",
    "eval_recovery",
    "RecoveryMetrics",
    "eval_observation_drift",
    "eval_trajectory_coherence",
    "rollout_coherence",
    "CoherenceMetrics",
    "eval_controllability",
    "ControllabilityMetrics",
]
