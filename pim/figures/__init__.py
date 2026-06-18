"""pim.figures — figure builders. Pure: take pre-computed arrays/metrics, return Figure.

No model calls, no metric computation. Notebooks/scripts compute metrics with
pim.eval, then pass the arrays here for visualisation.

Layout:
  theme.py            — palette + style_ax / style_ax_dark + plot_color
  setup.py            — training curves + dataset overview
  prediction.py       — 1-step MSE vs context, horizon RMSE
  recovery.py         — recovery bars, RMSE vs context, trajectory viz
  rollout.py          — observation drift, position drift, coherence,
                        rollout trajectory, 3-panel waterfall
  controllability.py  — per-step RMSE, position trajectory, waterfall triptych
"""

from pim.figures.theme import (
    PALETTE,
    plot_color,
    style_ax,
    style_ax_dark,
)
from pim.figures.setup import plot_dataset_overview, plot_training_curves
from pim.figures.prediction import plot_horizon_rmse, plot_mse_by_context
from pim.figures.recovery import (
    plot_recovery_bars,
    plot_recovery_by_context,
    plot_recovery_trajectory,
)
from pim.figures.rollout import (
    plot_coherence_bar,
    plot_coherence_distribution,
    plot_observation_drift,
    plot_position_drift,
    plot_rollout_3panel,
    plot_rollout_trajectory,
)
from pim.figures.controllability import (
    plot_controllability_obs,
    plot_controllability_positions,
    plot_controllability_trajectory,
    plot_controllability_waterfalls,
)

__all__ = [
    # theme
    "PALETTE",
    "plot_color",
    "style_ax",
    "style_ax_dark",
    # setup
    "plot_training_curves",
    "plot_dataset_overview",
    # prediction
    "plot_mse_by_context",
    "plot_horizon_rmse",
    # recovery
    "plot_recovery_bars",
    "plot_recovery_by_context",
    "plot_recovery_trajectory",
    # rollout
    "plot_observation_drift",
    "plot_position_drift",
    "plot_coherence_bar",
    "plot_coherence_distribution",
    "plot_rollout_trajectory",
    "plot_rollout_3panel",
    # controllability
    "plot_controllability_obs",
    "plot_controllability_positions",
    "plot_controllability_trajectory",
    "plot_controllability_waterfalls",
]
