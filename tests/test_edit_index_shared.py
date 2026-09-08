"""The Edit Index formula lives once (``pim.metrics.edit_index``) — both constructions
must reproduce their pre-2026-09-07 standalone implementations exactly."""

from __future__ import annotations

import numpy as np

from pim.metrics.set_editability import edit_index_legal, move_rmse, uniform_over_legal
from pim.metrics.zone_editability import _index_from


def _old_edit_index_legal(probs, legal_pre, legal_post, support="union"):
    out = np.full(len(probs), np.nan)
    for i, (L0, L1) in enumerate(zip(legal_pre, legal_post)):
        s0, s1 = set(L0), set(L1)
        idx = np.array(sorted(s0 | s1 if support == "union" else s0 ^ s1), int)
        if idx.size == 0:
            continue
        g0, g1 = uniform_over_legal(L0, probs.shape[1]), uniform_over_legal(L1, probs.shape[1])
        d_un = float(np.sqrt(((probs[i, idx] - g0[idx]) ** 2).mean()))
        d_ed = float(np.sqrt(((probs[i, idx] - g1[idx]) ** 2).mean()))
        if d_un + d_ed == 0:
            continue
        out[i] = (d_un - d_ed) / (d_un + d_ed)
    return out


def _old_index_from(pred, gt_edit, gt_uned, mask):
    out = np.full(len(pred), np.nan)
    for i in range(len(pred)):
        m = mask[i]
        if not m.any():
            continue
        d_e = np.sqrt(((pred[i, m] - gt_edit[i, m]) ** 2).mean())
        d_u = np.sqrt(((pred[i, m] - gt_uned[i, m]) ** 2).mean())
        if d_u + d_e > 1e-12:
            out[i] = (d_u - d_e) / (d_u + d_e)
    return float(np.nanmean(out))


def test_set_construction_matches_its_old_standalone_form():
    rng = np.random.default_rng(0)
    n, V = 300, 64
    probs = rng.dirichlet(np.ones(V) * 0.3, size=n).astype(np.float32)
    pre = [sorted(rng.choice(V, rng.integers(0, 12), replace=False).tolist()) for _ in range(n)]
    post = [sorted(rng.choice(V, rng.integers(0, 12), replace=False).tolist()) for _ in range(n)]
    post[5] = pre[5]                                   # identical sets: symdiff support empty
    for support in ("union", "symdiff"):
        new, old = edit_index_legal(probs, pre, post, support), _old_edit_index_legal(probs, pre, post, support)
        assert np.array_equal(np.isnan(new), np.isnan(old))
        assert np.allclose(new[~np.isnan(new)], old[~np.isnan(old)], atol=1e-12)
    ref = np.stack([uniform_over_legal(L, V) for L in post])
    old_rmse = np.nanmean([np.sqrt(((probs[i] - ref[i]) ** 2).mean()) for i, L in enumerate(post) if L])
    assert abs(move_rmse(probs, post) - old_rmse) < 1e-12


def test_zone_construction_matches_its_old_standalone_form():
    rng = np.random.default_rng(1)
    n, R = 200, 128
    pred, gt_e, gt_u = (rng.random((n, R)).astype(np.float32) for _ in range(3))
    mask = rng.random((n, R)) < 0.2
    mask[3] = False                                    # an unscoreable case
    # the old zone path took the ratio in float32; the shared formula takes it in float64
    # (the set path always did) — identical to float32 rounding, ~1e-7 on the index
    assert abs(_index_from(pred, gt_e, gt_u, mask) - _old_index_from(pred, gt_e, gt_u, mask)) < 1e-6
