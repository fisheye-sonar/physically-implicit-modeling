"""Qualitative + quantitative rollout comparison: GRU vs RSSM (vs DiT) vs ground truth.

Built for the RSSM-refinement completion deliverable. Near-equal horizon MSE can
hide a real generative-quality gap: a deterministic prior-MEAN rollout can win on
MSE by *smearing* (hedging) rather than committing to a sharp object position.
So we look in observation space, not just at the MSE table.

Produces (into --out):
  horizon_curve.png      — clean-obs MSE vs rollout step: GRU, RSSM(mean),
                           RSSM(sampled), [DiT(mean), DiT(sampled)], persistence.
  waterfalls.png         — per-sample obs-space waterfalls (time x scan-position):
                           GT | GRU | RSSM(mean) | RSSM(sampled) [| DiT(mean) |
                           DiT(sampled)], rollout boundary marked.
  sharpness.txt          — total-variation sharpness of the rollout region
                           (blurry/smeared rollouts have lower TV than GT).

Usage
-----
    python scripts/compare_rollouts.py \
        --gru runs/gru/3_dset3_gru_persistentids_inview_400epochs/best_model.pt \
        --rssm runs/rssm_sweep2/FINAL/latest.pt \
        --dit runs/dit/0_dset4_dit_baseline/best_model.pt \
        --data-dir datasets/4_fixed_refl_inview --out runs/rssm_sweep2/figs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pim.eval._helpers import autoregressive_rollout, autoregressive_rollouts  # noqa: E402
from pim.world_models import load_checkpoint, load_dataset  # noqa: E402


def _rollout_set(model, obs, n_context, device, sample=None, predict_mode=None):
    """AR rollout obs for many samples; set the model's mode toggles if applicable.

    sample       — RSSM stochastic toggle (model.sample)
    predict_mode — DiT deterministic-readout toggle (model.predict_mode)
    """
    if sample is not None and hasattr(model, "sample"):
        model.sample = sample
    if predict_mode is not None and hasattr(model, "predict_mode"):
        model.predict_mode = predict_mode
    return autoregressive_rollouts(model, obs, n_context=n_context, device=device)


def _tv(frames: np.ndarray) -> float:
    """Mean total-variation (spatial gradient magnitude) per frame — sharpness."""
    return float(np.abs(np.diff(frames, axis=-1)).sum(axis=-1).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gru", required=True)
    p.add_argument("--rssm", required=True)
    p.add_argument("--dit", default=None, help="optional DiT checkpoint to include")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-context", type=int, default=10)
    p.add_argument("--n-roll-mse", type=int, default=2000)
    p.add_argument("--samples", type=int, nargs="+", default=[0, 1, 2, 3])
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    test = load_dataset(a.data_dir, n_obj_keep=2, require_edits=False).test
    T = test.T_frames
    nc = a.n_context
    n_roll = T - nc

    gru, _ = load_checkpoint(a.gru, device=a.device)
    rssm, _ = load_checkpoint(a.rssm, device=a.device)
    dit = load_checkpoint(a.dit, device=a.device)[0] if a.dit else None

    # ── Horizon-MSE curves (clean obs) ──
    clean = test.clean_obs[:a.n_roll_mse]
    obs_in = test.obs[:a.n_roll_mse]
    gru_roll = _rollout_set(gru, obs_in, nc, a.device)
    rssm_mean = _rollout_set(rssm, obs_in, nc, a.device, sample=False)
    rssm_samp = _rollout_set(rssm, obs_in, nc, a.device, sample=True)
    dit_mean = dit_samp = None
    if dit is not None:
        dit_mean = _rollout_set(dit, obs_in, nc, a.device, predict_mode="mean")
        dit_samp = _rollout_set(dit, obs_in, nc, a.device, predict_mode="sample")
    tgt = clean[:, nc:nc + n_roll, :]
    persist = np.repeat(obs_in[:, nc - 1:nc, :], n_roll, axis=1)

    def hmse(roll):
        return ((roll - tgt) ** 2).mean(axis=(0, 2))

    steps = np.arange(1, n_roll + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, hmse(gru_roll), label="GRU", lw=2)
    ax.plot(steps, hmse(rssm_mean), label="RSSM (prior mean)", lw=2)
    ax.plot(steps, hmse(rssm_samp), label="RSSM (sampled)", lw=2, ls="--")
    if dit is not None:
        ax.plot(steps, hmse(dit_mean), label="DiT (mean)", lw=2)
        ax.plot(steps, hmse(dit_samp), label="DiT (sampled)", lw=2, ls="--")
    ax.plot(steps, hmse(persist), label="persistence", lw=1.5, color="gray", ls=":")
    ax.set_xlabel("rollout step (open-loop)")
    ax.set_ylabel("clean-obs MSE")
    ax.set_title("Fig 1 — open-loop horizon MSE (lower = better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "horizon_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Waterfalls: GT | GRU | RSSM(mean) | RSSM(sampled) [| DiT(mean) | DiT(sampled)] ──
    cols = ["GT (clean)", "GRU", "RSSM (mean)", "RSSM (sampled)"]
    if dit is not None:
        cols += ["DiT (mean)", "DiT (sampled)"]
    nrows = len(a.samples)
    fig, axes = plt.subplots(
        nrows, len(cols), figsize=(3.25 * len(cols), 3 * nrows), squeeze=False
    )
    vmax = float(np.percentile(test.clean_obs[:50], 99.5))
    for r, idx in enumerate(a.samples):
        gt_full = test.clean_obs[idx]  # (T,R)
        gr, _ = autoregressive_rollout(gru, test.obs[idx], nc, a.device)
        if hasattr(rssm, "sample"):
            rssm.sample = False
        rm, _ = autoregressive_rollout(rssm, test.obs[idx], nc, a.device)
        if hasattr(rssm, "sample"):
            rssm.sample = True
        rs, _ = autoregressive_rollout(rssm, test.obs[idx], nc, a.device)
        # stitch: real context + predicted rollout
        ctx = test.obs[idx, :nc]
        panels = [
            gt_full,
            np.concatenate([ctx, gr], axis=0),
            np.concatenate([ctx, rm], axis=0),
            np.concatenate([ctx, rs], axis=0),
        ]
        if dit is not None:
            dit.predict_mode = "mean"
            dm, _ = autoregressive_rollout(dit, test.obs[idx], nc, a.device)
            dit.predict_mode = "sample"
            ds, _ = autoregressive_rollout(dit, test.obs[idx], nc, a.device)
            panels += [
                np.concatenate([ctx, dm], axis=0),
                np.concatenate([ctx, ds], axis=0),
            ]
        for c, (title, panel) in enumerate(zip(cols, panels)):
            ax = axes[r][c]
            ax.imshow(panel, aspect="auto", origin="upper", cmap="magma",
                      vmin=0, vmax=vmax, interpolation="nearest")
            ax.axhline(nc - 0.5, color="cyan", lw=1.0, ls="--")
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(f"sample {idx}\ntime →")
            ax.set_xticks([])
    fig.suptitle("Fig 2 — observation-space rollout (time × scan position); "
                 "cyan = rollout start", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "waterfalls.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Sharpness (total variation over the rollout region) ──
    lines = [
        "Total-variation sharpness over the open-loop rollout region "
        "(higher = sharper edges; smeared/blurry = lower):",
        f"  GT (clean)      : {_tv(tgt):.4f}",
        f"  GRU             : {_tv(gru_roll):.4f}",
        f"  RSSM (mean)     : {_tv(rssm_mean):.4f}",
        f"  RSSM (sampled)  : {_tv(rssm_samp):.4f}",
    ]
    if dit is not None:
        lines += [
            f"  DiT (mean)      : {_tv(dit_mean):.4f}",
            f"  DiT (sampled)   : {_tv(dit_samp):.4f}",
        ]
    lines += [
        "",
        f"near-h MSE (H1-5): GRU={hmse(gru_roll)[:5].mean():.5f}  "
        f"RSSM_mean={hmse(rssm_mean)[:5].mean():.5f}  "
        f"RSSM_samp={hmse(rssm_samp)[:5].mean():.5f}  "
        + (
            f"DiT_mean={hmse(dit_mean)[:5].mean():.5f}  "
            f"DiT_samp={hmse(dit_samp)[:5].mean():.5f}  "
            if dit is not None
            else ""
        )
        + f"persist={hmse(persist)[:5].mean():.5f}",
    ]
    (out / "sharpness.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved figures + sharpness.txt → {out}")


if __name__ == "__main__":
    main()
