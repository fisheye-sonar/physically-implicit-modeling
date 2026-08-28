"""Probe Skill must agree with R² on the regression branch and behave on the classification one."""
from __future__ import annotations

import numpy as np
import pytest

from pim.eval import (
    probe_skill_classification,
    probe_skill_regression,
    trivial_error_rate,
)
from pim.extractors.standard import _r2


def test_regression_branch_is_exactly_r2():
    """It must not be a second, subtly different R² — every existing R² IS a Probe Skill."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(500, 3))
    pred = y + 0.3 * rng.normal(size=y.shape)
    mu = rng.normal(size=3) * 0.1
    assert probe_skill_regression(pred, y, mu) == pytest.approx(_r2(pred, y, mu), abs=1e-12)


def test_regression_perfect_and_trivial():
    y = np.array([[1.0], [2.0], [3.0]])
    mu = np.array([2.0])
    assert probe_skill_regression(y, y, mu) == pytest.approx(1.0)
    assert probe_skill_regression(np.full_like(y, 2.0), y, mu) == pytest.approx(0.0)


def test_regression_worse_than_trivial_is_negative():
    y = np.array([[1.0], [2.0], [3.0]])
    assert probe_skill_regression(np.full_like(y, 10.0), y, np.array([2.0])) < 0


def test_regression_constant_target_raises_instead_of_dividing_by_zero():
    y = np.full((10, 1), 5.0)
    with pytest.raises(ValueError, match="constant"):
        probe_skill_regression(y, y, np.array([5.0]))


def test_majority_is_taken_per_output_dimension():
    """A board whose squares have different majorities must not be collapsed to one."""
    train = np.array([[0, 1]] * 9 + [[1, 0]])       # dim0 majority 0, dim1 majority 1
    held = np.array([[0, 1], [0, 1], [1, 1]])
    assert trivial_error_rate(held, train) == pytest.approx(1 / 6)
    # collapsing to a single global majority (0) would give 3/6 — guard against that regression
    assert trivial_error_rate(held, train) != pytest.approx(3 / 6)


def test_majority_comes_from_train_not_heldout():
    """A baseline fitted on the evaluation split flatters every score computed against it."""
    train = np.array([[0]] * 10
                     )
    held = np.array([[1]] * 10)
    assert trivial_error_rate(held, train) == pytest.approx(1.0)   # train majority 0, all wrong


def test_classification_perfect_and_trivial():
    train = np.array([[0]] * 9 + [[1]])
    held = np.array([[0], [0], [1], [1]])
    assert probe_skill_classification(held, held, train) == pytest.approx(1.0)
    majority = np.zeros_like(held)
    assert probe_skill_classification(majority, held, train) == pytest.approx(0.0)


def test_classification_worse_than_majority_is_negative():
    train = np.array([[0]] * 9 + [[1]])
    held = np.array([[0], [0], [0], [1]])
    always_wrong = np.array([[1], [1], [1], [0]])
    assert probe_skill_classification(always_wrong, held, train) < 0


def test_classification_shape_mismatch_raises():
    with pytest.raises(ValueError, match="!="):
        probe_skill_classification(np.zeros((4, 1), int), np.zeros((4, 2), int),
                                   np.zeros((4, 2), int))


def test_classification_undefined_when_majority_is_already_perfect():
    train = held = np.zeros((5, 1), dtype=int)
    with pytest.raises(ValueError, match="undefined"):
        probe_skill_classification(held, held, train)
