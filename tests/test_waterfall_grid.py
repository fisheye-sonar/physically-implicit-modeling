"""Tests for the canonical waterfall (`pim.figures.waterfall_grid`).

These lock in the spec rules that are enforced structurally rather than by documentation —
see `notebooks/experiments/editability/WATERFALL_SPEC.md`. The point of the helper is that the
recurring violations become impossible or loud, so these tests guard exactly that.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from pim.figures import waterfall_grid

N, K, W, N_CTX = 4, 15, 128, 6


def _arrays(seed: int = 0):
    rng = np.random.default_rng(seed)
    ctx = rng.random((N, N_CTX, W))
    gt = rng.random((N, K, W))
    cols = {
        "Unsteered": rng.random((N, K, W)),
        "Injection": rng.random((N, K, W)),
    }
    return cols, ctx, gt


def test_gt_column_is_first_and_every_method_gets_a_column():
    cols, ctx, gt = _arrays()
    fig = waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1, 2])
    # 3 sample rows x (GT + 2 methods)
    assert len(fig.axes) == 3 * 3
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles[0].startswith("GT")
    assert any("Unsteered" in t for t in titles)
    assert any("Injection" in t for t in titles)


def test_fixed_scaling_is_applied_to_every_cell():
    """Per-cell autoscaling makes a collapsed arm look normal — the failure the panel exists
    to catch. Every image must carry the same explicit limits."""
    cols, ctx, gt = _arrays()
    # One arm collapsed to a constant: with autoscaling this would render as full contrast.
    cols["Collapsed"] = np.full((N, K, W), 0.5)
    fig = waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1, 2])
    images = [im for ax in fig.axes for im in ax.get_images()]
    assert images, "no image cells drawn"
    for im in images:
        assert im.get_clim() == (0.0, 1.0)


def test_context_frames_sit_above_each_columns_own_body():
    """The banned pattern is a shared teacher-forced row painted across all columns. The API
    makes it unrepresentable: context is shared, the body is per-column. Verify the composed
    panel is context-then-that-column's-own-rollout."""
    cols, ctx, gt = _arrays()
    fig = waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1, 2])
    images = [im for ax in fig.axes for im in ax.get_images()]
    panel = images[1].get_array()  # row 0, first method column
    assert panel.shape == (N_CTX + K, W)
    np.testing.assert_allclose(panel[:N_CTX], ctx[0])
    np.testing.assert_allclose(panel[N_CTX:], cols["Unsteered"][0])


def test_gray_colormap():
    cols, ctx, gt = _arrays()
    fig = waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1, 2])
    for ax in fig.axes:
        for im in ax.get_images():
            assert im.get_cmap().name == "gray"


def test_fewer_than_three_sample_rows_warns():
    cols, ctx, gt = _arrays()
    with pytest.warns(UserWarning, match="at least 3"):
        waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1])


def test_three_sample_rows_does_not_warn():
    cols, ctx, gt = _arrays()
    with warnings_as_errors():
        waterfall_grid(cols, ctx, gt, title="t", sample_idx=[0, 1, 2])


def test_metrics_and_leads_by_one_appear_in_column_titles():
    cols, ctx, gt = _arrays()
    fig = waterfall_grid(
        cols,
        ctx,
        gt,
        title="t",
        sample_idx=[0, 1, 2],
        metrics={"Injection": -0.51},
        leads_by_one=("Injection",),
    )
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    inj = next(t for t in titles if "Injection" in t)
    assert "-0.51" in inj
    assert "leads by one" in inj


def test_empty_columns_rejected():
    _, ctx, gt = _arrays()
    with pytest.raises(ValueError):
        waterfall_grid({}, ctx, gt, title="t")


class warnings_as_errors:
    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)
