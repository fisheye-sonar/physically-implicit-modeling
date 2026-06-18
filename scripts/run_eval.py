"""Run the full evaluation pipeline against a checkpoint + dataset directory.

Auto-detects model type from the checkpoint (GRU vs RSSM). Mirrors the
notebooks/{gru,rssm}_eval.ipynb step-by-step structure: load, predict,
recover, roll out, control. Saves all figures + metrics.json to a
timestamped output directory.

Usage:
    python scripts/run_eval.py \\
        --checkpoint runs/my_run/best_model.pt \\
        --data-dir   datasets/my_dataset \\
        --output-dir outputs/my_run \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

# Allow running from repo root without install
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "notebooks"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from helpers.nb_viz import (
    animate_3panel,
    animate_ar_gt_vs_predicted,
    animate_gt_vs_predicted,
)
from pim.eval import (
    autoregressive_rollout,
    autoregressive_rollouts,
    collect_rollouts,
    compute_obs_baselines,
    compute_pos_baselines,
    decode_states_multi,
    eval_controllability,
    eval_horizon_mse,
    eval_mse_by_context,
    eval_observation_drift,
    eval_position_controllability,
    eval_position_drift,
    eval_recovery_multi,
    eval_single_step,
    eval_trajectory_coherence,
    fit_probes,
    per_sample_coherence,
    rollout_steered,
    rollout_unsteered,
    teacher_force,
    warm_up_to_edit,
)
from pim.extractors import (
    LinearExtractor,
    MLPExtractor,
    ProbeSpec,
    StateDefinition,
    hungarian_mse,
    identity_mse,
)
from pim.figures import (
    plot_coherence_bar,
    plot_coherence_distribution,
    plot_controllability_obs,
    plot_controllability_positions,
    plot_controllability_trajectory,
    plot_controllability_waterfalls,
    plot_dataset_overview,
    plot_horizon_rmse,
    plot_mse_by_context,
    plot_observation_drift,
    plot_position_drift,
    plot_recovery_bars,
    plot_recovery_by_context,
    plot_recovery_trajectory,
    plot_rollout_3panel,
    plot_rollout_trajectory,
    plot_training_curves,
)
from pim.simulator.dataset import load_sample
from pim.simulator.viz import save_animation
from pim.world_models import load_checkpoint, load_dataset, make_test_loader


@dataclass
class EvalConfig:
    checkpoint: str
    data_dir: str
    output_dir: str
    device: str = "cuda"
    batch_size: int = 512
    num_workers: int = 6
    n_obj: int = 2
    use_hungarian: bool = False
    n_context_pred: int = 10
    n_context_roll: int = 20
    n_rollout: int = 20
    coherence_n_eval: int = 500
    ctrl_n_rollout: int = 15
    n_viz_roll: int = 3
    n_viz_ctrl: int = 3
    skip_animations: bool = False


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="World-model evaluation (GRU + RSSM auto-detect)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", required=True, dest="data_dir",
                   help="Directory with dataset.json + test.h5 + edits.h5")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", default=512, type=int, dest="batch_size")
    p.add_argument("--num-workers", default=6, type=int, dest="num_workers")
    p.add_argument("--n-obj", default=2, type=int, dest="n_obj")
    p.add_argument("--use-hungarian", action="store_true", dest="use_hungarian")
    p.add_argument("--n-context-pred", default=10, type=int, dest="n_context_pred")
    p.add_argument("--n-context-roll", default=20, type=int, dest="n_context_roll")
    p.add_argument("--n-rollout", default=20, type=int, dest="n_rollout")
    p.add_argument("--coherence-n-eval", default=500, type=int, dest="coherence_n_eval")
    p.add_argument("--ctrl-n-rollout", default=15, type=int, dest="ctrl_n_rollout")
    p.add_argument("--n-viz-roll", default=3, type=int, dest="n_viz_roll")
    p.add_argument("--n-viz-ctrl", default=3, type=int, dest="n_viz_ctrl")
    p.add_argument("--skip-animations", action="store_true", dest="skip_animations")
    args = p.parse_args()

    if args.output_dir is None:
        run_name = Path(args.checkpoint).parent.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(Path("outputs") / "eval" / run_name / timestamp)
    return EvalConfig(**{k: v for k, v in vars(args).items()})


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Figure] = {}

    print(f"\nOutput → {out_dir}\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    model, ckpt_info = load_checkpoint(cfg.checkpoint, device=cfg.device)
    bundle = load_dataset(cfg.data_dir, n_obj_keep=cfg.n_obj)
    test, edits = bundle.test, bundle.edits
    test_loader = make_test_loader(test, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    obs_baselines = compute_obs_baselines(test.obs, test.clean_obs, test.obs_noise_std)
    pos_baselines = compute_pos_baselines(test.positions, test.position_noise_std)

    is_rssm = "det_size" in ckpt_info.model_config
    print(f"Model    : {ckpt_info.run_name}  ({'RSSM' if is_rssm else 'GRU'})")
    print(f"Epoch    : {ckpt_info.epoch}   val_loss={ckpt_info.val_loss:.6f}")
    print(f"Dataset  : {test.n_samples:,} samples  T={test.T_frames}  R={test.obs_res}")

    figs["training_linlog"] = plot_training_curves(
        ckpt_info.metrics_history, best_epoch=ckpt_info.epoch,
        run_name=ckpt_info.run_name, show_components=is_rssm, log_x=False,
    )
    figs["training_loglog"] = plot_training_curves(
        ckpt_info.metrics_history, best_epoch=ckpt_info.epoch,
        run_name=ckpt_info.run_name, show_components=is_rssm, log_x=True,
    )
    figs["dataset_overview"] = plot_dataset_overview(test.h5_path, n_samples=8)

    # ── Predictive Quality ────────────────────────────────────────────────────
    print("\n[1/4] Predictive Quality")
    preds_tf, states_tf = teacher_force(model, test_loader, device=cfg.device)
    preds_ar = autoregressive_rollouts(model, test.obs, n_context=cfg.n_context_pred, device=cfg.device)
    prediction_metrics = eval_single_step(test.obs, preds_tf)
    horizon_mse_noisy = eval_horizon_mse(test.obs, preds_ar, cfg.n_context_pred)
    horizon_mse_clean = eval_horizon_mse(test.clean_obs, preds_ar, cfg.n_context_pred)
    context_lengths, mse_by_ctx_noisy = eval_mse_by_context(model, test_loader, device=cfg.device)
    mse_by_ctx_clean = ((preds_tf - test.clean_obs[:, 1:, :]) ** 2).mean(axis=(0, 2))
    print(f"  next-step MSE: {prediction_metrics.mean_mse:.6f} ± {prediction_metrics.std_mse:.6f}")

    figs["mse_by_context"] = plot_mse_by_context(
        context_lengths, mse_by_ctx_noisy, mse_by_ctx_clean,
        n_context_warmup=cfg.n_context_pred, baselines=obs_baselines,
    )
    figs["horizon_rmse"] = plot_horizon_rmse(
        horizon_mse_noisy, horizon_mse_clean,
        n_context=cfg.n_context_pred, baselines=obs_baselines,
    )

    # ── Recovery ──────────────────────────────────────────────────────────────
    print("\n[2/4] Recovery")
    state_def = StateDefinition(
        name="positions", state_shape=(cfg.n_obj, 2),
        extract_fn=lambda batch: batch["positions"],
    )
    env_states_tf = test.positions[:, :-1, :cfg.n_obj, :]
    vis_mask_tf = test.is_visible[:, :-1, :cfg.n_obj].all(axis=2)
    loss_fn = hungarian_mse if cfg.use_hungarian else identity_mse

    probes = [
        ProbeSpec(
            name="linear",
            probe=LinearExtractor(model.hidden_size, state_def, use_lstsq=True),
            marker="x", color_idx=0, linestyle="-",
        ),
        ProbeSpec(
            name="MLP",
            probe=MLPExtractor(model.hidden_size, state_def, mlp_hidden=256, n_epochs=30, lr=5e-3),
            marker="o", color_idx=1, linestyle="--",
        ),
    ]
    fit_probes(probes, states_tf, env_states_tf,
               mask=vis_mask_tf, loss_fn=loss_fn, device=cfg.device)
    recovery_metrics = eval_recovery_multi(
        probes, states_tf, env_states_tf,
        mask=vis_mask_tf, use_hungarian=cfg.use_hungarian, device=cfg.device,
    )
    for name, m in recovery_metrics.items():
        print(f"  {name:8s} overall MSE = {m.overall_mse:.6f}")

    figs["recovery_bars"] = plot_recovery_bars(recovery_metrics, probes, n_obj=cfg.n_obj)
    figs["recovery_by_context"] = plot_recovery_by_context(recovery_metrics, probes, baselines=pos_baselines)

    import h5py
    with h5py.File(test.h5_path, "r") as f:
        colors_first = f["colors"][:3, :cfg.n_obj]
    decoded_tf_viz = decode_states_multi(probes, states_tf[:3], device=cfg.device)
    for i in range(3):
        figs[f"recovery_traj_{i}"] = plot_recovery_trajectory(
            positions_gt=test.positions[i, :-1, :cfg.n_obj],
            decoded_per_probe={p.name: decoded_tf_viz[p.name][i] for p in probes},
            probes=probes, scene_colors=colors_first[i],
            vis_mask=vis_mask_tf[i], sample_idx=i, title_suffix="teacher forcing",
        )

    # ── Rollout Consistency ───────────────────────────────────────────────────
    print("\n[3/4] Rollout Consistency")
    _, h_roll, obs_roll = collect_rollouts(
        model, test.obs[:cfg.coherence_n_eval],
        n_context=cfg.n_context_roll, n_rollout=cfg.n_rollout, device=cfg.device,
    )
    decoded_roll = decode_states_multi(probes, h_roll, device=cfg.device)
    drift_mse_noisy = eval_observation_drift(test.obs[:cfg.coherence_n_eval], obs_roll, cfg.n_context_roll)
    drift_mse_clean = eval_observation_drift(test.clean_obs[:cfg.coherence_n_eval], obs_roll, cfg.n_context_roll)
    position_drift = {
        p.name: eval_position_drift(decoded_roll[p.name], test.positions[:cfg.coherence_n_eval], cfg.n_context_roll)
        for p in probes
    }
    gt_window = test.positions[:cfg.coherence_n_eval, cfg.n_context_roll:cfg.n_context_roll + cfg.n_rollout, :cfg.n_obj]
    gt_scores = per_sample_coherence(gt_window)
    probe_scores = {p.name: per_sample_coherence(decoded_roll[p.name]) for p in probes}
    coherence = {p.name: eval_trajectory_coherence(decoded_roll[p.name]) for p in probes}
    print(f"  GT       smoothness mean = {gt_scores.mean():.4f}")
    for p in probes:
        print(f"  {p.name:8s} smoothness mean = {coherence[p.name].mean_score:.4f}")

    figs["observation_drift"] = plot_observation_drift(
        drift_mse_noisy, drift_mse_clean,
        n_context=cfg.n_context_roll, n_rollout=cfg.n_rollout, baselines=obs_baselines,
    )
    figs["position_drift"] = plot_position_drift(
        position_drift, probes,
        n_context=cfg.n_context_roll, n_rollout=cfg.n_rollout, baselines=pos_baselines,
    )
    figs["coherence_bar"] = plot_coherence_bar(gt_scores, probe_scores, probes)
    figs["coherence_distribution"] = plot_coherence_distribution(gt_scores, probe_scores, probes)

    for i in range(min(cfg.n_viz_roll, cfg.coherence_n_eval)):
        scene_i, *_ = load_sample(test.h5_path, i)
        decoded_i = {p.name: decoded_roll[p.name][i] for p in probes}
        sample_scores = {"GT": float(gt_scores[i])}
        sample_scores.update({p.name: float(probe_scores[p.name][i]) for p in probes})
        figs[f"rollout_traj_{i}"] = plot_rollout_trajectory(
            positions_gt=test.positions[i, :, :cfg.n_obj],
            decoded_per_probe=decoded_i, probes=probes, scene_colors=scene_i.colors,
            sample_idx=i, n_context=cfg.n_context_roll, n_rollout=cfg.n_rollout,
            sample_scores=sample_scores,
        )
        score_str = "  ".join(f"{k}={v:.3f}" for k, v in sample_scores.items())
        figs[f"rollout_3panel_{i}"] = plot_rollout_3panel(
            test_h5_path=test.h5_path,
            positions_gt=test.positions[i, :, :cfg.n_obj],
            obs_rollout=obs_roll[i], decoded_per_probe=decoded_i, probes=probes,
            sample_idx=i, n_context=cfg.n_context_roll, n_rollout=cfg.n_rollout,
            suptitle=f"Sample {i}  —  rollout smoothness  {score_str}",
        )

    # ── Counterfactual Controllability ────────────────────────────────────────
    print("\n[4/4] Counterfactual Controllability")
    linear_probe = next(p.probe for p in probes if isinstance(p.probe, LinearExtractor))
    N_CTRL = min(500, edits.n_samples)
    warm = warm_up_to_edit(
        model, edits.obs[:N_CTRL], edits.edit_frame,
        n_viz=cfg.n_viz_ctrl, n_ctx_show=8, device=cfg.device,
    )
    targets = edits.positions[:N_CTRL, edits.edit_frame, :cfg.n_obj, :].reshape(N_CTRL, cfg.n_obj * 2)
    steered = rollout_steered(model, warm.h_at_edit, targets, linear_probe,
                              n_rollout=cfg.ctrl_n_rollout, device=cfg.device)
    unsteered = rollout_unsteered(model, warm.h_at_edit,
                                  n_rollout=cfg.ctrl_n_rollout, device=cfg.device)
    gt_obs = edits.obs[:N_CTRL, edits.edit_frame:edits.edit_frame + cfg.ctrl_n_rollout]
    clean_gt_obs = edits.clean_obs[:N_CTRL, edits.edit_frame:edits.edit_frame + cfg.ctrl_n_rollout]
    gt_positions = edits.positions[:N_CTRL, edits.edit_frame:edits.edit_frame + cfg.ctrl_n_rollout, :cfg.n_obj, :]
    ctrl_metrics = eval_controllability(steered, unsteered, gt_obs, clean_gt_obs)
    ctrl_pos_rmse = eval_position_controllability(steered, unsteered, gt_positions, probes, device=cfg.device)
    print(f"  steered MSE   = {ctrl_metrics.steered_mse:.6f}")
    print(f"  unsteered MSE = {ctrl_metrics.unsteered_mse:.6f}")
    print(f"  injection err = {ctrl_metrics.injection_error:.6f}")

    figs["ctrl_obs"] = plot_controllability_obs(ctrl_metrics, baselines=obs_baselines)
    figs["ctrl_pos"] = plot_controllability_positions(ctrl_pos_rmse, probes, baselines=pos_baselines)

    pre_decoded = decode_states_multi(probes, warm.h_pre_edit, device=cfg.device)
    s_decoded = decode_states_multi(probes, steered.h[:cfg.n_viz_ctrl], device=cfg.device)
    u_decoded = decode_states_multi(probes, unsteered.h[:cfg.n_viz_ctrl], device=cfg.device)
    for i in range(min(cfg.n_viz_ctrl, N_CTRL)):
        pre_gt = edits.positions[i, edits.edit_frame - warm.n_ctx_show:edits.edit_frame, :cfg.n_obj]
        figs[f"ctrl_traj_{i}"] = plot_controllability_trajectory(
            pre_edit_gt=pre_gt,
            pre_edit_decoded={p.name: pre_decoded[p.name][i] for p in probes},
            post_edit_gt=gt_positions[i],
            steered_decoded={p.name: s_decoded[p.name][i] for p in probes},
            probes=probes, scene_colors=edits.colors[i, :cfg.n_obj],
            sample_idx=i, edit_frame=edits.edit_frame, n_rollout=cfg.ctrl_n_rollout,
            show_unsteered=True,
            unsteered_decoded={p.name: u_decoded[p.name][i] for p in probes},
        )
        figs[f"ctrl_waterfall_{i}"] = plot_controllability_waterfalls(
            pre_edit_obs=edits.obs[i, :edits.edit_frame],
            gt_post_obs=gt_obs[i], steered_obs=steered.obs[i], unsteered_obs=unsteered.obs[i],
            sample_idx=i, edit_frame=edits.edit_frame, n_rollout=cfg.ctrl_n_rollout,
        )

    # ── Persist figures + metrics + config ────────────────────────────────────
    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved {len(figs)} figures → {out_dir}")

    metrics = {
        "prediction": {
            "next_step_mse_mean": prediction_metrics.mean_mse,
            "next_step_mse_std": prediction_metrics.std_mse,
        },
        "recovery": {name: m.overall_mse for name, m in recovery_metrics.items()},
        "rollout": {
            "gt_smoothness_mean": float(gt_scores.mean()),
            **{f"{name}_smoothness_mean": float(c.mean_score) for name, c in coherence.items()},
            **{f"{name}_jump_ratio_mean": float(c.mean_jump_ratio) for name, c in coherence.items()},
        },
        "controllability": {
            "steered_mse": ctrl_metrics.steered_mse,
            "unsteered_mse": ctrl_metrics.unsteered_mse,
            "injection_error": ctrl_metrics.injection_error,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "eval_config.json").write_text(json.dumps({
        "eval_config": asdict(cfg),
        "checkpoint": {
            "epoch": ckpt_info.epoch,
            "val_loss": ckpt_info.val_loss,
            "model_config": ckpt_info.model_config,
        },
    }, indent=2))

    # ── Animations (one per model: GRU does 2, RSSM does 1) ──────────────────
    if cfg.skip_animations:
        print(f"\nDone. Results in {out_dir}")
        return

    print("\nGenerating animations...")
    scene, obs_depth, obs_id, obs_intensity = load_sample(test.h5_path, 0)

    if is_rssm:
        pred, _ = autoregressive_rollout(model, obs_intensity, cfg.n_context_pred, device=cfg.device)
        anim = animate_3panel(
            scene, obs_depth, obs_id, obs_intensity,
            pred, cfg.n_context_pred,
            title=f"Sample 0  |  warm-up={cfg.n_context_pred}  (RSSM imagination)",
            dark=True,
        )
        save_animation(anim, str(out_dir / "rssm_imagination.gif"), fps=12)
    else:
        # GRU: TF animation
        with torch.no_grad():
            h0 = torch.from_numpy(states_tf[0:1]).float().to(cfg.device)
            decoded_pos = linear_probe(h0).cpu().numpy()[0]
        anim = animate_gt_vs_predicted(
            scene, obs_depth, obs_id, obs_intensity, decoded_pos,
            title="Sample 0  |  GT (solid) vs decoded positions (dashed)",
            dark=True,
        )
        save_animation(anim, str(out_dir / "gt_vs_decoded.gif"), fps=12)

        # GRU: AR animation
        n_ctx_ar = test.T_frames // 2
        pred_ar, states_ar = autoregressive_rollout(model, obs_intensity, n_ctx_ar, device=cfg.device)
        with torch.no_grad():
            decoded_pos_ar = linear_probe(torch.from_numpy(states_ar).float().to(cfg.device)).cpu().numpy()
        anim_ar = animate_ar_gt_vs_predicted(
            scene, obs_depth, obs_id, obs_intensity,
            pred_ar, decoded_pos_ar, n_context=n_ctx_ar,
            title=f"Sample 0  |  AR rollout from frame {n_ctx_ar}",
            dark=True,
        )
        save_animation(anim_ar, str(out_dir / "gt_vs_decoded_ar.gif"), fps=12)

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
