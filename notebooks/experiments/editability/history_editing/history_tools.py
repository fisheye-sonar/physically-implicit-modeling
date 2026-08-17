"""Shared helpers for the `history_editing` thread — drawing + small numerics.

Two notebooks (`gru_history_editing.ipynb`, `transformer_history_editing.ipynb`) test the same
hypothesis on two architectures, so the pieces that must be **identical** between them live here:

* `waterfall_grid` — the ONE observation-space waterfall, baking in the whole fixed spec from
  `../../../CLAUDE.md` (gray on dark, GT column, noisy context frames above a marked edit line,
  each column its own free-run from step 0, green/red locators, figure-top legend). Per that file:
  do NOT re-implement the panel per notebook — that is where the drift happens.
* the lag-probe numerics, which are the notebooks' central measurement and must be computed the
  same way on both architectures to be comparable.

Pipeline logic (which arms, in what order, with what targets) stays **in the notebooks** —
`CLAUDE.md` § "Notebook = orchestrator".
"""

from __future__ import annotations

import numpy as np

# ── the fixed waterfall spec ──────────────────────────────────────────────────

DARK, TXT, EDGE = "#0a0a14", "#a3adc2", "#fa8850"
GREEN, RED = "#00E676", "#FF5252"
N_CTX = 6  # pre-edit context frames shown above the edit line


def ray_centroid(mask: np.ndarray) -> np.ndarray:
    """Mean ray index of each row of a boolean (N, R) zone mask; NaN where empty."""
    out = np.full(len(mask), np.nan)
    for i in range(len(mask)):
        idx = np.where(mask[i])[0]
        if idx.size:
            out[i] = idx.mean()
    return out


def waterfall_grid(
    *,
    rolls: dict[str, np.ndarray],
    ctx: np.ndarray,
    gt_roll: np.ndarray,
    tgt_cx: np.ndarray,
    ghost_cx: np.ndarray,
    samples,
    edit_frame: int,
    title: str,
    labels: dict[str, str] | None = None,
    leads_by_one: tuple[str, ...] = ("Oracle observation",),
    col_width: float = 2.5,
    row_height: float = 3.2,
):
    """The canonical 1-D observation waterfall (`CLAUDE.md` § Waterfalls).

    Parameters
    ----------
    rolls      : {arm name -> (N, K, R)} each arm's OWN free-run, step 0 first.
    ctx        : (N, N_CTX, R) the **noisy** observations the model was teacher-forced on,
                 `edits.obs[:, ef-N_CTX:ef]` — NOT the clean render (only the GT column is clean).
    gt_roll    : (N, K, R) the sim's CLEAN post-edit observations, `clean_obs[ef:ef+K]`.
    tgt_cx     : (N,) ray centroid of the target zone   → solid green locator.
    ghost_cx   : (N,) ray centroid of the ghost zone    → dashed red locator.
    samples    : row indices to draw (one notebook sample per row).
    labels     : {arm name -> column title}, normally carrying that arm's headline metric.
    leads_by_one : arms fed `obs[ef]` and therefore one frame ahead — LABELLED, never re-aligned.

    Alignment (get this exactly right): `warm_up_to_edit` teacher-forces `obs[0..ef-1]`, so a
    rollout's step 0 decodes frame `ef`. Hence `rolls[a][:, 0:K]` is plotted directly against
    `gt_roll = clean_obs[ef:ef+K]` — no slicing, no dropped step.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    labels = labels or {}
    cols = ["GT (sim clean obs)"] + list(rolls.keys())
    K = gt_roll.shape[1]
    fig, axes = plt.subplots(
        len(samples),
        len(cols),
        figsize=(col_width * len(cols), row_height * len(samples)),
        squeeze=False,
        facecolor=DARK,
    )
    for r, smp in enumerate(samples):
        for c, name in enumerate(cols):
            ax = axes[r][c]
            ax.set_facecolor(DARK)
            body = gt_roll[smp] if c == 0 else rolls[name][smp][:K]
            panel = np.clip(np.concatenate([ctx[smp], body], 0), 0, 1)
            ax.imshow(
                panel,
                aspect="auto",
                origin="upper",
                cmap="gray",  # FIXED — never magma/viridis
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.axhline(N_CTX - 0.5, color=EDGE, lw=1.2, ls="--", alpha=0.9)
            if not np.isnan(tgt_cx[smp]):
                ax.axvline(tgt_cx[smp], color=GREEN, lw=1.4)
            if not np.isnan(ghost_cx[smp]):
                ax.axvline(ghost_cx[smp], color=RED, ls="--", lw=1.4)
            if r == 0:
                lab = labels.get(name, name)
                if name in leads_by_one:
                    lab += "\n(leads by one frame)"
                ax.set_title(lab, fontsize=7.5, color=(GREEN if c == 0 else TXT))
            if c == 0:
                ax.set_ylabel(f"sample {smp}\nsim frame", fontsize=8, color=TXT)
                ax.set_yticks([0, N_CTX, N_CTX + K // 2, N_CTX + K - 1])
                ax.set_yticklabels(
                    [
                        edit_frame - N_CTX,
                        edit_frame,
                        edit_frame + K // 2,
                        edit_frame + K - 1,
                    ],
                    fontsize=7,
                    color=TXT,
                )
            else:
                ax.set_yticks([])
            ax.set_xticks([])
    handles = [
        Line2D(
            [0],
            [0],
            color=GREEN,
            lw=2,
            label="target location (where the object should go)",
        ),
        Line2D(
            [0], [0], color=RED, ls="--", lw=2, label="ghost location (where it was)"
        ),
        Line2D(
            [0],
            [0],
            color=EDGE,
            ls="--",
            lw=2,
            label=(
                f"edit applied here — the {N_CTX} rows above are the NOISY observations the model "
                "was teacher-forced on; every row below is that column's OWN free-run"
            ),
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=1,
        fontsize=8.5,
        frameon=False,
        labelcolor=TXT,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(title, y=1.055, fontsize=11, color=TXT)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


# ── the lag-probe measurement (identical on both architectures) ───────────────


def _fit_lstsq(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)[0]


def _apply(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.c_[X, np.ones(len(X))] @ A


def r2_vs_train_mean(pred: np.ndarray, y: np.ndarray, train_mean: np.ndarray) -> float:
    """R² against the TRAIN mean — the held-out baseline a probe has to beat."""
    return float(1 - ((pred - y) ** 2).sum() / ((y - train_mean) ** 2).sum())


def lag_probe_curve(
    states: np.ndarray,  # (N, T, H) states, index t aligned to frame t
    pos: np.ndarray,  # (N, T_full, D) true positions, flat per frame
    vel: np.ndarray,  # (N, T_full, D) true velocities, flat per frame
    *,
    lags,
    dt: float,
    t_min: int,
    holdout: float = 0.2,
    seed: int = 0,
) -> dict:
    """Held-out linear R² for `state_t → pos(t−k)`, against the two baselines that decide
    whether the curve means anything.

    Returns, per lag `k`:
      `direct`    — a linear probe fit straight to `pos(t−k)`.
      `ballistic` — **the no-stored-history null**: fit ONE probe `state → (pos_t, v_t)`, then
                    extrapolate `p̂(t−k) = p̂os_t − k·dt·v̂_t`. Nothing about the past is read.
      `ceiling`   — the same extrapolation from the TRUE `(pos_t, v_t)`; the most any
                    history-free model could score, given the world's position noise.
      `shuffled`  — labels permuted across sequences; the floor.

    `direct − ballistic` is the load-bearing quantity: it is the readable information about the
    past that is **not** already implied by the present state and velocity. The split is by
    SEQUENCE (never by row — consecutive frames are near-duplicates).
    """
    n_seq, T = states.shape[0], states.shape[1]
    n_tr = int(round((1 - holdout) * n_seq))
    rng = np.random.default_rng(seed)

    X = states[:, t_min:T].astype(np.float64)
    Xtr = X[:n_tr].reshape(-1, X.shape[-1])
    Xte = X[n_tr:].reshape(-1, X.shape[-1])

    PV = np.concatenate([pos[:, t_min:T], vel[:, t_min:T]], -1).astype(np.float64)
    D = pos.shape[-1]
    PVtr, PVte = PV[:n_tr].reshape(-1, 2 * D), PV[n_tr:].reshape(-1, 2 * D)

    A_pv = _fit_lstsq(Xtr, PVtr)
    PV_hat = _apply(A_pv, Xte)  # the model's own (pos, vel) readout, held out

    out = {k: {} for k in ("direct", "ballistic", "ceiling", "shuffled")}
    out["posvel_r2"] = {
        "pos": r2_vs_train_mean(PV_hat[:, :D], PVte[:, :D], PVtr.mean(0)[:D]),
        "vel": r2_vs_train_mean(PV_hat[:, D:], PVte[:, D:], PVtr.mean(0)[D:]),
    }
    for k in lags:
        Ytr = pos[:n_tr, t_min - k : T - k].reshape(-1, D).astype(np.float64)
        Yte = pos[n_tr:, t_min - k : T - k].reshape(-1, D).astype(np.float64)
        mu = Ytr.mean(0)
        out["direct"][k] = r2_vs_train_mean(_apply(_fit_lstsq(Xtr, Ytr), Xte), Yte, mu)
        out["ballistic"][k] = r2_vs_train_mean(
            PV_hat[:, :D] - k * dt * PV_hat[:, D:], Yte, mu
        )
        out["ceiling"][k] = r2_vs_train_mean(
            _apply(_fit_lstsq(PVtr, Ytr), PVte), Yte, mu
        )
        perm = rng.permutation(len(Xtr))
        out["shuffled"][k] = r2_vs_train_mean(
            _apply(_fit_lstsq(Xtr, Ytr[perm]), Xte), Yte, mu
        )
    return out


def stacked_lag_probe(
    states: np.ndarray,  # (N, T, H)
    pos: np.ndarray,  # (N, T_full, D)
    *,
    n_lags: int,
    t_min: int,
    n_train_seq: int | None = None,
):
    """Fit `A_n : state → [pos(t), pos(t−1), …, pos(t−n)]` in ONE least-squares solve.

    Returns `(W, b, sv)` with `W` of shape `(D·(n+1), H)` and `sv` its singular values.

    ⚠ The spectrum is the point. In a world whose velocity is constant,
    `pos(t−k) = pos(t) − k·dt·v(t)` up to the per-step position noise, so the least-squares
    block rows satisfy `A_k ≈ A_pos − k·dt·A_vel`: the row space is dominated by an
    **8-dimensional** `(pos, vel)` core no matter how large `n` is, and whatever genuinely
    encodes the past shows up only in the far weaker trailing singular directions. Report the
    spectrum, and compute any chance level from the EFFECTIVE rank rather than from `D·(n+1)`.
    """
    T = states.shape[1]
    n_tr = n_train_seq or states.shape[0]
    X = states[:n_tr, t_min:T].reshape(-1, states.shape[-1]).astype(np.float64)
    Y = (
        np.concatenate([pos[:n_tr, t_min - k : T - k] for k in range(n_lags + 1)], -1)
        .reshape(-1, pos.shape[-1] * (n_lags + 1))
        .astype(np.float64)
    )
    A = _fit_lstsq(X, Y)
    W, b = A[:-1].T, A[-1]
    return W, b, np.linalg.svd(W, compute_uv=False)


def effective_rank(sv: np.ndarray, tol: float = 1e-2) -> int:
    """# singular values above `tol × sv[0]` — the rank that actually carries the write.

    A numeric rank (`tol=1e-6`) counts directions whose pseudoinverse gain is ~10⁶ and which no
    real edit can use; the chance level for a subspace fraction must be computed from the rank
    that carries energy, not from that one.
    """
    return int((sv > tol * sv[0]).sum())


def subspace_fraction(vecs: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Per-row ‖P_basis v‖/‖v‖ for an orthonormal `basis` of shape (rank, H)."""
    proj = vecs @ basis.T @ basis
    return np.linalg.norm(proj, axis=1) / np.maximum(
        np.linalg.norm(vecs, axis=1), 1e-12
    )


def chance_fraction(rank: int, H: int) -> float:
    """Expected ‖P v‖/‖v‖ for a random v in R^H projected onto a `rank`-dim subspace.

    `√(rank/H)` — the level a "small" projection fraction must be read against. `CLAUDE.md`:
    when the rank varies across a comparison, plot the **enrichment** `value/chance`, never the
    raw fraction, or the moving chance level manufactures a trend on its own.
    """
    return float(np.sqrt(rank / H))
