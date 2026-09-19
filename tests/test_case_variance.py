"""Case-level spread beside every arm (2026-09-18): the helpers are deterministic, drop NaN cases
the way the scalars do, and the scorecards carry them without changing any existing number."""
import numpy as np

from pim.figures.tables import ci95_halfwidth, t975
from pim.metrics.edit_index import case_stats, ratio_ci95
from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio, move_scorecard


def test_case_stats_matches_the_scalar_and_drops_nan():
    v = np.array([0.2, 0.4, np.nan, 0.6, 0.8])
    s = case_stats(v, prefix="ei_")
    assert s["ei_n_cases"] == 4
    assert abs(s["ei_case_sd"] - np.std([0.2, 0.4, 0.6, 0.8], ddof=1)) < 1e-12
    assert abs(s["ei_case_se"] - s["ei_case_sd"] / 2) < 1e-12
    assert s["ei_ci95_lo"] <= np.nanmean(v) <= s["ei_ci95_hi"]
    assert case_stats(v) == case_stats(v)                       # a fixed seed: reproducible
    assert np.isnan(case_stats([0.5])["case_sd"]) and case_stats([0.5])["n_cases"] == 1


def test_ratio_ci95_brackets_the_ratio_and_pairs_cases():
    rng = np.random.default_rng(1)
    den = rng.uniform(0.5, 1.5, 400)
    num = 0.7 * den                                             # every case has ratio 0.7 exactly
    lo, hi = ratio_ci95(num, den)
    assert abs(lo - 0.7) < 1e-9 and abs(hi - 0.7) < 1e-9        # paired resampling: no spread
    lo, hi = ratio_ci95(num ** 2, den ** 2, root=True)
    assert abs(lo - 0.7) < 1e-9 and abs(hi - 0.7) < 1e-9
    num2 = num.copy()
    num2[:5] = np.nan
    lo, hi = ratio_ci95(num2, den)
    assert abs(lo - 0.7) < 1e-9


def test_move_scorecard_gains_case_fields_without_moving_the_scalars():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(64), size=20).astype(np.float32)
    pre = [[int(x) for x in rng.choice(64, 5, replace=False)] for _ in range(20)]
    post = [sorted(set(L[:3] + [int(y) for y in rng.choice(64, 2, replace=False)])) for L in pre]
    c = move_scorecard(probs, pre, post)
    assert abs(c["edit_index_union"] - np.nanmean(c["edit_index_union_per_case"])) < 1e-6
    assert c["edit_index_union_n_cases"] == 20 and c["edit_index_symdiff_n_cases"] <= 20
    assert len(c["rmse_post_per_case"]) == 20
    ci = move_fidelity_ci95(probs, probs, post)
    assert abs(ci["fidelity_ci95_lo"] - 1.0) < 1e-9 and abs(ci["fidelity_ci95_hi"] - 1.0) < 1e-9
    assert move_fidelity_ratio(probs, probs, post) == 1.0


def test_t_halfwidth():
    assert t975(2) == 4.303 and t975(100) == 1.960 and t975(22) == 2.086
    v = [0.50, 0.54, 0.52]
    assert abs(ci95_halfwidth(v) - 4.303 * np.std(v, ddof=1) / np.sqrt(3)) < 1e-12
    assert np.isnan(ci95_halfwidth([0.5]))
