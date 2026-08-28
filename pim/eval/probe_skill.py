"""Probe Skill — one comparable axis for probe quality across regression and classification.

Motivation
----------
This project reads world state out of two settings whose probes are scored in different units:
discworld positions/velocities are a **regression** (reported as R²), Othello board state is a
**classification** (reported by Li et al. as an error rate, %). Those cannot share a figure axis,
which is why decodability across the two settings has never been plotted together.

Probe Skill removes the unit::

    skill = 1 - probe_loss / trivial_predictor_loss

Both branches then mean the same thing:

* **1.0** — the probe is perfect.
* **0.0** — the probe is no better than the trivial predictor that ignores the representation
  entirely (predict the train mean; predict the majority class).
* **< 0** — the probe is *worse* than knowing nothing, which on held-out data means it fitted
  its own training set. See the `MLP < linear` tripwire in `pim.extractors.standard`.

For regression against the train mean this is **exactly R²** — deliberately, so every R² already
in the repo is already a Probe Skill and nothing needs recomputing. The classification branch is
the new part.

⚠ The trivial predictor must come from the **train** split, never the held-out one. A baseline
fitted on the evaluation data is itself a probe with free information, and it flatters every
score computed against it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probe_skill_regression", "probe_skill_classification", "trivial_error_rate"]


def probe_skill_regression(pred: np.ndarray, y: np.ndarray,
                           train_mean: np.ndarray) -> float:
    """1 − SSE(probe) / SSE(train-mean predictor). Identical to R² against the train mean."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    denom = ((y - np.asarray(train_mean, float)) ** 2).sum()
    if denom <= 0:
        raise ValueError("trivial predictor has zero error — the target is constant on this "
                         "split, so skill is undefined (any probe is trivially perfect)")
    return float(1.0 - ((pred - y) ** 2).sum() / denom)


def trivial_error_rate(y_true: np.ndarray, train_labels: np.ndarray,
                       n_classes: int | None = None) -> float:
    """Error rate of the majority-class predictor, with the majority taken from TRAIN.

    `y_true` (N, D) held-out integer labels, `train_labels` (M, D) the train split. The majority
    is taken **per output dimension** — for a 64-square Othello board the majority class of a
    corner and of a centre square are different, and collapsing them understates the baseline.
    """
    y_true = np.asarray(y_true)
    train_labels = np.asarray(train_labels)
    if y_true.ndim == 1:
        y_true, train_labels = y_true[:, None], train_labels[:, None]
    if y_true.shape[1] != train_labels.shape[1]:
        raise ValueError(f"held-out has {y_true.shape[1]} output dims, "
                         f"train has {train_labels.shape[1]}")
    k = n_classes or int(max(y_true.max(), train_labels.max())) + 1
    maj = np.array([np.bincount(train_labels[:, d], minlength=k).argmax()
                    for d in range(train_labels.shape[1])])
    return float((y_true != maj[None, :]).mean())


def probe_skill_classification(pred_labels: np.ndarray, y_true: np.ndarray,
                               train_labels: np.ndarray,
                               n_classes: int | None = None) -> float:
    """1 − err(probe) / err(majority-class predictor), majority taken from TRAIN."""
    pred_labels, y_true = np.asarray(pred_labels), np.asarray(y_true)
    if pred_labels.shape != y_true.shape:
        raise ValueError(f"pred {pred_labels.shape} != truth {y_true.shape}")
    base = trivial_error_rate(y_true, train_labels, n_classes)
    if base <= 0:
        raise ValueError("majority-class predictor is already perfect on this split — skill is "
                         "undefined (there is nothing for a probe to add)")
    return float(1.0 - (pred_labels != y_true).mean() / base)
