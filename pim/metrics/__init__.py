"""pim.metrics — canonical scoring, arrays in, numbers out. Never imports matplotlib.

Three modules, one per question:

    decodability.py   can a probe read the state out?  Probe Skill (the cross-environment
                      axis: 1 = perfect, 0 = trivial baseline) + native R² / error rate.
    editability.py    did an edit land, on discworld?  ray-zone RMSEs, the Edit Index,
                      fidelity_ratio, the edit scorecard.
    othello_moves.py  did an edit land, on Othello?    Li error, legal mass, the
                      legal-set Edit Index (``edit_index_legal``), the move scorecard,
                      and the guard (``move_fidelity_ratio``).

THE GUARD, one definition and one polarity in both environments (2026-09-01):
``RMSE(edited prediction, edited-world GT) / RMSE(unsteered prediction, same GT)``,
evaluated on the **edit step only** — ``fidelity_ratio`` on discworld,
``move_fidelity_ratio`` on Othello. **> 1 = degraded, not steered.** It is the absolute
counterpart to the Edit Index, which is relative and so cannot see a wrecked output that
happens to land marginally nearer the edited world.

The two Edit Index constructions share the formula and the axis but not the ingredients,
so they keep distinct names and modules — quote which one you mean.

Every metric here has a registry row in ``research/REGISTRY.md``. Import these; never
re-derive the formulas at a call site.
"""

from pim.metrics.decodability import (
    probe_skill_classification,
    probe_skill_regression,
    r2,
    trivial_error_rate,
)
from pim.metrics.editability import (
    DIFF_EPS,
    SCORECARD_COLUMNS,
    EditZones,
    build_edit_zones,
    direction_report,
    edit_index,
    edit_index_by_step,
    edit_scorecard,
    fidelity_ratio,
    object_constants,
    random_samples,
    representative_samples,
    shift_zones,
    sim_config_from,
    zone_rmse,
)
from pim.metrics.othello_moves import (
    N_TILES,
    edit_index_legal,
    li_error,
    move_fidelity_ratio,
    move_rmse,
    move_scorecard,
    uniform_over_legal,
)

__all__ = [
    # decodability
    "probe_skill_regression",
    "probe_skill_classification",
    "trivial_error_rate",
    "r2",
    # discworld editability
    "DIFF_EPS",
    "EditZones",
    "SCORECARD_COLUMNS",
    "build_edit_zones",
    "direction_report",
    "edit_index",
    "edit_index_by_step",
    "edit_scorecard",
    "fidelity_ratio",
    "object_constants",
    "random_samples",
    "representative_samples",
    "shift_zones",
    "sim_config_from",
    "zone_rmse",
    # othello editability
    "N_TILES",
    "edit_index_legal",
    "li_error",
    "move_scorecard",
    "move_rmse",
    "move_fidelity_ratio",
    "uniform_over_legal",
]
