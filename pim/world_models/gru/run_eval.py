"""End-to-end GRU evaluation pipeline.

Implements all four evaluation criteria as pure functions over typed result
dataclasses. Designed for two calling contexts:

  Notebook (interactive):
      from pim.eval.run import EvalConfig, setup, run_criterion1, plot_criterion1
      cfg = EvalConfig(...)
      s   = setup(cfg)
      c1  = run_criterion1(cfg, s)
      for fig in plot_criterion1(cfg, s, c1).values():
          display(fig); plt.close()

  CLI (batch):
      # scripts/gru_eval.py calls run_all(cfg) then save_figures / save_metrics

Design: run_* functions print progress and return typed dataclasses.
plot_* functions return dict[str, Figure] — caller decides show vs. save.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pim.eval._helpers import collect_rollout, run_autoregressive, run_teacher_forcing
from pim.eval.controllability import ControllabilityMetrics, eval_controllability
from pim.eval.plotting import (
    PALETTE,
    plot_coherence_distribution,
    plot_horizon_sweep,
    plot_mse_by_context as _plot_mse_by_context,
    plot_per_component_bars,
    plot_trajectory_comparison,
    plot_training_curves as _plot_training_curves,
    style_ax,
)
from pim.eval.prediction import (
    PredictionMetrics,
    eval_horizon_mse,
    eval_mse_by_context,
    eval_single_step,
)
from pim.eval.recovery import RecoveryMetrics, eval_recovery
from pim.eval.rollout import (
    CoherenceMetrics,
    eval_observation_drift,
    eval_trajectory_coherence,
    rollout_coherence,
)
from pim.extractors.base import StateDefinition
from pim.extractors.linear import LinearExtractor
from pim.extractors.matching import hungarian_mse, identity_mse
from pim.extractors.mlp import MLPExtractor
from pim.extractors.training import fit_lstsq, train_extractor
from pim.world_models.dataloader import ObservationDataset
from pim.world_models.gru import GRUModel, ModelConfig


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class EvalConfig:
    """All knobs for one evaluation run."""

    checkpoint_path: str
    test_h5_path: str
    edits_h5_path: str
    output_dir: str = "outputs/eval"
    device: str = "cuda"
    batch_size: int = 512
    num_workers: int = 6

    # Criterion 1 — Predictive Quality
    n_context: int = 10

    # Criterion 2 — Recovery
    n_obj: int = 2
    use_hungarian: bool = False
    use_lstsq: bool = True
    probe_n_epochs: int = 30
    probe_lr: float = 5e-3
    probe_hidden_dim: int = 256   # MLP hidden layer width

    # Criterion 3 — Rollout Consistency
    rollout_n_context: int = 20
    rollout_n_rollout: int = 20
    coherence_n_eval: int = 500

    # Criterion 4 — Controllability
    ctrl_n_rollout: int = 15

    # Which criteria to run
    criteria: tuple[int, ...] = (1, 2, 3, 4)


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class SetupResult:
    model: GRUModel
    ckpt_info: dict
    test_loader: DataLoader
    obs_actual: np.ndarray      # (N, T, R)
    positions_gt: np.ndarray    # (N, T, n_obj, 2)
    is_visible: np.ndarray      # (N, T, n_obj)
    run_name: str
    T_frames: int
    obs_res: int
    metrics_history: list[dict]


@dataclass
class C1Result:
    """Criterion 1 — Predictive Quality."""
    metrics: PredictionMetrics
    obs_pred_tf: np.ndarray         # (N, T-1, R)  teacher-forcing predictions
    internal_states_tf: np.ndarray  # (N, T-1, H)  hidden states
    obs_rollout: np.ndarray         # (N, T-n_context, R)  AR predictions
    horizon_mse: np.ndarray         # (T-n_context,)  MSE at each horizon step
    context_lengths: np.ndarray     # (T-1,)
    mse_by_ctx: np.ndarray          # (T-1,)  1-step MSE as function of context


@dataclass
class C2Result:
    """Criterion 2 — Recovery."""
    linear_extractor: LinearExtractor
    mlp_extractor: MLPExtractor
    recovery_linear: RecoveryMetrics
    recovery_mlp: RecoveryMetrics
    env_states_tf: np.ndarray   # (N, T-1, n_obj, 2)  GT states aligned to h_tf
    vis_mask_tf: np.ndarray     # (N, T-1)  both-visible mask


@dataclass
class C3Result:
    """Criterion 3 — Rollout Consistency."""
    drift_mse: np.ndarray              # (n_rollout,)
    coherence_metrics: CoherenceMetrics
    decoded_pos_roll: np.ndarray       # (N_eval, n_rollout, n_obj, 2)
    per_sample_scores: np.ndarray      # (N_eval,)


@dataclass
class C4Result:
    """Criterion 4 — Counterfactual Controllability."""
    ctrl_metrics: ControllabilityMetrics
    h_at_edit: np.ndarray          # (N, H)
    env_state_targets: np.ndarray  # (N, n_obj*2)
    obs_at_edit: np.ndarray        # (N, R)
    obs_post_edit: np.ndarray      # (N, T-edit_frame, R)
    edit_frame: int


# ── Setup ─────────────────────────────────────────────────────────────────────


def setup(cfg: EvalConfig) -> SetupResult:
    """Load model checkpoint and test dataset."""
    ckpt = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model = GRUModel(ModelConfig(**ckpt["model_config"])).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ckpt_info = {
        "epoch":        ckpt["epoch"],
        "val_loss":     ckpt["val_loss"],
        "model_config": ckpt["model_config"],
        "train_config": ckpt["train_config"],
    }

    metrics_path = Path(cfg.checkpoint_path).parent / "metrics.jsonl"
    metrics_history = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]

    with h5py.File(cfg.test_h5_path, "r") as f:
        T_frames     = f["obs_intensity"].shape[1]
        obs_res      = f["obs_intensity"].shape[2]
        obs_actual   = f["obs_intensity"][:].astype(np.float32)
        positions_gt = f["positions"][:, :, :cfg.n_obj, :].astype(np.float32)
        is_visible   = f["is_visible"][:, :, :cfg.n_obj].astype(bool)
        n_samples    = f["obs_intensity"].shape[0]

    ds = ObservationDataset(
        cfg.test_h5_path,
        np.arange(n_samples),
        keys=("obs_intensity", "positions", "is_visible"),
    )
    test_loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.num_workers > 0),
        persistent_workers=(cfg.num_workers > 0),
    )

    print(f"Model    : {Path(cfg.checkpoint_path).parent.name}")
    print(f"Epoch    : {ckpt_info['epoch']}   val_loss={ckpt_info['val_loss']:.6f}")
    print(f"Dataset  : {n_samples:,} samples  T={T_frames}  R={obs_res}")

    return SetupResult(
        model=model,
        ckpt_info=ckpt_info,
        test_loader=test_loader,
        obs_actual=obs_actual,
        positions_gt=positions_gt,
        is_visible=is_visible,
        run_name=Path(cfg.checkpoint_path).parent.name,
        T_frames=T_frames,
        obs_res=obs_res,
        metrics_history=metrics_history,
    )


# ── Criterion 1: Predictive Quality ──────────────────────────────────────────


def run_criterion1(cfg: EvalConfig, s: SetupResult) -> C1Result:
    print("Criterion 1 — Predictive Quality")

    print("  teacher-forcing inference...")
    obs_pred_tf, internal_states_tf = run_teacher_forcing(
        s.model, s.test_loader, cfg.device,
    )
    metrics = eval_single_step(s.obs_actual, obs_pred_tf)
    print(f"  next-step MSE: {metrics.mean_mse:.6f} ± {metrics.std_mse:.6f}")

    print("  AR rollout (all test samples)...")
    all_rollouts = []
    for i in tqdm(range(len(s.obs_actual)), desc="  AR rollout", leave=False):
        rollout, _ = run_autoregressive(s.model, s.obs_actual[i], cfg.n_context, cfg.device)
        all_rollouts.append(rollout)
    obs_rollout = np.stack(all_rollouts)
    horizon_mse = eval_horizon_mse(s.obs_actual, obs_rollout, cfg.n_context)
    print(f"  horizon MSE  step 1={horizon_mse[0]:.4f}  step {len(horizon_mse)}={horizon_mse[-1]:.4f}")

    print("  MSE by context length...")
    context_lengths, mse_by_ctx = eval_mse_by_context(
        s.model, s.test_loader, n_steps_ahead=1, device=cfg.device,
    )

    return C1Result(
        metrics=metrics,
        obs_pred_tf=obs_pred_tf,
        internal_states_tf=internal_states_tf,
        obs_rollout=obs_rollout,
        horizon_mse=horizon_mse,
        context_lengths=context_lengths,
        mse_by_ctx=mse_by_ctx,
    )


# ── Criterion 2: Recovery ─────────────────────────────────────────────────────


def run_criterion2(cfg: EvalConfig, s: SetupResult, c1: C1Result) -> C2Result:
    print("Criterion 2 — Recovery")

    state_def = StateDefinition(
        name="positions",
        state_shape=(cfg.n_obj, 2),
        extract_fn=lambda batch: batch["positions"],
    )
    env_states_tf = s.positions_gt[:, :-1, :cfg.n_obj, :]       # (N, T-1, n_obj, 2)
    vis_mask_tf   = s.is_visible[:, :-1, :cfg.n_obj].all(axis=2)  # (N, T-1)
    loss_fn = hungarian_mse if cfg.use_hungarian else identity_mse

    print("  training LinearExtractor...")
    linear_extractor = LinearExtractor(s.model.hidden_size, state_def)
    if cfg.use_lstsq:
        mse = fit_lstsq(
            linear_extractor, c1.internal_states_tf, env_states_tf, mask=vis_mask_tf,
        )
        print(f"  lstsq train MSE: {mse:.6f}")
    else:
        losses = train_extractor(
            linear_extractor, c1.internal_states_tf, env_states_tf,
            n_epochs=cfg.probe_n_epochs, lr=cfg.probe_lr,
            loss_fn=loss_fn, mask=vis_mask_tf, device=cfg.device,
        )
        print(f"  final train loss: {losses[-1]:.6f}")

    print("  training MLPExtractor...")
    mlp_extractor = MLPExtractor(s.model.hidden_size, state_def, hidden_dim=cfg.probe_hidden_dim)
    mlp_losses = train_extractor(
        mlp_extractor, c1.internal_states_tf, env_states_tf,
        n_epochs=cfg.probe_n_epochs, lr=cfg.probe_lr,
        loss_fn=loss_fn, mask=vis_mask_tf, device=cfg.device,
    )
    print(f"  MLP final train loss: {mlp_losses[-1]:.6f}")

    recovery_linear = eval_recovery(
        env_states_tf, c1.internal_states_tf, linear_extractor,
        mask=vis_mask_tf, use_hungarian=cfg.use_hungarian, device=cfg.device,
    )
    recovery_mlp = eval_recovery(
        env_states_tf, c1.internal_states_tf, mlp_extractor,
        mask=vis_mask_tf, use_hungarian=cfg.use_hungarian, device=cfg.device,
    )
    print(f"  Linear overall MSE: {recovery_linear.overall_mse:.6f}")
    print(f"  MLP    overall MSE: {recovery_mlp.overall_mse:.6f}")

    return C2Result(
        linear_extractor=linear_extractor,
        mlp_extractor=mlp_extractor,
        recovery_linear=recovery_linear,
        recovery_mlp=recovery_mlp,
        env_states_tf=env_states_tf,
        vis_mask_tf=vis_mask_tf,
    )


# ── Criterion 3: Rollout Consistency ─────────────────────────────────────────


def run_criterion3(cfg: EvalConfig, s: SetupResult, c2: C2Result) -> C3Result:
    print("Criterion 3 — Rollout Consistency")
    print(f"  collecting {cfg.coherence_n_eval} rollouts...")

    all_h_roll, all_obs_roll = [], []
    for i in tqdm(range(cfg.coherence_n_eval), desc="  rollouts", leave=False):
        _, h_roll, obs_roll = collect_rollout(
            s.model, s.obs_actual[i],
            cfg.rollout_n_context, cfg.rollout_n_rollout, cfg.device,
        )
        all_h_roll.append(h_roll)
        all_obs_roll.append(obs_roll)

    h_rollout_arr  = np.stack(all_h_roll)   # (N_eval, n_rollout, H)
    obs_rollout_co = np.stack(all_obs_roll)  # (N_eval, n_rollout, R)

    drift_mse = eval_observation_drift(
        s.obs_actual[:cfg.coherence_n_eval], obs_rollout_co, cfg.rollout_n_context,
    )

    c2.linear_extractor.eval()
    with torch.no_grad():
        h_t = torch.from_numpy(h_rollout_arr).float().to(cfg.device)
        decoded_pos_roll = c2.linear_extractor(h_t).cpu().numpy()

    coherence_metrics = eval_trajectory_coherence(decoded_pos_roll)
    per_sample_scores = np.array([
        rollout_coherence(decoded_pos_roll[i])[0]
        for i in range(cfg.coherence_n_eval)
    ])
    print(f"  coherence mean={coherence_metrics.mean_score:.4f}  jump ratio={coherence_metrics.mean_jump_ratio:.2f}")

    return C3Result(
        drift_mse=drift_mse,
        coherence_metrics=coherence_metrics,
        decoded_pos_roll=decoded_pos_roll,
        per_sample_scores=per_sample_scores,
    )


# ── Criterion 4: Controllability ──────────────────────────────────────────────


def run_criterion4(cfg: EvalConfig, s: SetupResult, c2: C2Result) -> C4Result:
    print("Criterion 4 — Counterfactual Controllability")

    with h5py.File(cfg.edits_h5_path, "r") as f:
        edit_frames     = f["edit_frame"][:].astype(int)
        edits_obs       = f["obs_intensity"][:].astype(np.float32)
        edits_positions = f["positions"][:].astype(np.float32)

    edit_frame = int(edit_frames[0])
    n_ctrl = min(500, edits_obs.shape[0])
    print(f"  edit_frame={edit_frame}  n_ctrl={n_ctrl}")

    h_at_edit = np.zeros((n_ctrl, s.model.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(n_ctrl), desc="  teacher-force to edit frame", leave=False):
            obs_seq = torch.from_numpy(edits_obs[i]).float().to(cfg.device)
            h = None
            for t in range(edit_frame):
                _, h = s.model.step(obs_seq[t].unsqueeze(0), h)
            h_at_edit[i] = h[0, 0].cpu().numpy()

    env_state_targets = edits_positions[:n_ctrl, edit_frame, :cfg.n_obj, :].reshape(n_ctrl, cfg.n_obj * 2)
    obs_at_edit   = edits_obs[:n_ctrl, edit_frame, :]
    obs_post_edit = edits_obs[:n_ctrl, edit_frame:, :]

    ctrl_metrics = eval_controllability(
        internal_states_at_edit=h_at_edit,
        env_state_targets=env_state_targets,
        extractor=c2.linear_extractor,
        model=s.model,
        obs_post_edit_actual=obs_post_edit,
        obs_at_edit=obs_at_edit,
        n_rollout=cfg.ctrl_n_rollout,
        device=cfg.device,
    )
    ratio = ctrl_metrics.unsteered_mse / (ctrl_metrics.steered_mse + 1e-12)
    print(f"  steered={ctrl_metrics.steered_mse:.6f}  unsteered={ctrl_metrics.unsteered_mse:.6f}  ratio={ratio:.2f}x")

    return C4Result(
        ctrl_metrics=ctrl_metrics,
        h_at_edit=h_at_edit,
        env_state_targets=env_state_targets,
        obs_at_edit=obs_at_edit,
        obs_post_edit=obs_post_edit,
        edit_frame=edit_frame,
    )


# ── Plot functions — return dict[str, Figure] ─────────────────────────────────


def plot_setup(cfg: EvalConfig, s: SetupResult) -> dict[str, Figure]:
    """Lin-log and log-log training curves."""
    figs: dict[str, Figure] = {}

    figs["training_linlog"] = _plot_training_curves(
        s.metrics_history,
        title=f"{s.run_name}  —  training curves",
        log_y=True,
    )

    epochs     = [m["epoch"] for m in s.metrics_history]
    train_loss = [m["train_loss"] for m in s.metrics_history]
    val_loss   = [m["val_loss"]   for m in s.metrics_history]
    best_epoch = s.ckpt_info["epoch"]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#ffffff")
    style_ax(ax)
    ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=1.8, label="train")
    ax.plot(epochs, val_loss,   color=PALETTE[1], linewidth=1.8, label="val", linestyle="--")
    ax.axvline(best_epoch, color="#555555", linewidth=1.0, linestyle=":", alpha=0.7,
               label=f"best epoch {best_epoch}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch (log)", fontsize=10)
    ax.set_ylabel("loss (log)", fontsize=10)
    ax.set_title(f"{s.run_name}  —  training curves (log-log)", fontsize=11)
    ax.legend(frameon=False)
    plt.tight_layout()
    figs["training_loglog"] = fig

    return figs


def plot_criterion1(cfg: EvalConfig, s: SetupResult, c1: C1Result) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}

    figs["horizon_mse"] = plot_horizon_sweep(
        np.arange(1, len(c1.horizon_mse) + 1),
        c1.horizon_mse,
        title=f"Next-n observation MSE  (warm-up={cfg.n_context})",
    )
    figs["mse_by_context"] = _plot_mse_by_context(
        c1.context_lengths,
        c1.mse_by_ctx,
        title="Observation MSE vs context length  (1-step prediction)",
    )
    return figs


def plot_criterion2(
    cfg: EvalConfig,
    s: SetupResult,
    c1: C1Result,
    c2: C2Result,
    n_traj: int = 3,
) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}
    comp_labels = [f"obj{i} {d}" for i in range(cfg.n_obj) for d in ("x", "y")]

    figs["recovery_per_component"] = plot_per_component_bars(
        comp_labels,
        {"Linear": c2.recovery_linear.per_component_mse,
         "MLP":    c2.recovery_mlp.per_component_mse},
        title="Per-component position recovery MSE",
    )

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#ffffff")
    style_ax(ax)
    ax.plot(np.arange(1, len(c2.recovery_linear.mse_by_context) + 1),
            c2.recovery_linear.mse_by_context, color=PALETTE[0], linewidth=1.8, label="Linear")
    ax.plot(np.arange(1, len(c2.recovery_mlp.mse_by_context) + 1),
            c2.recovery_mlp.mse_by_context, color=PALETTE[1], linewidth=1.8, label="MLP",
            linestyle="--")
    ax.set_xlabel("context frames", fontsize=10)
    ax.set_ylabel("position recovery MSE", fontsize=10)
    ax.set_title("Position recovery MSE vs context length", fontsize=11)
    ax.legend(frameon=False)
    plt.tight_layout()
    figs["recovery_by_context"] = fig

    # Trajectory comparison: decoded positions vs GT
    c2.linear_extractor.eval()
    with torch.no_grad():
        h_t = torch.from_numpy(c1.internal_states_tf[:n_traj]).float().to(cfg.device)
        decoded_pos = c2.linear_extractor(h_t).cpu().numpy()

    with h5py.File(cfg.test_h5_path, "r") as f:
        colors_all = f["colors"][:n_traj, :cfg.n_obj]

    for i in range(n_traj):
        fig = plot_trajectory_comparison(
            s.positions_gt[i, :-1, :cfg.n_obj],
            {"Linear probe": decoded_pos[i]},
            scene_colors=colors_all[i],
            title=f"Sample {i}  —  position recovery (teacher forcing)",
        )
        figs[f"recovery_traj_{i}"] = fig

    return figs


def plot_criterion3(cfg: EvalConfig, s: SetupResult, c3: C3Result) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}
    comp_labels = [f"obj{i} {d}" for i in range(cfg.n_obj) for d in ("x", "y")]

    figs["observation_drift"] = plot_horizon_sweep(
        np.arange(1, len(c3.drift_mse) + 1),
        c3.drift_mse,
        title=f"Observation drift  (warm-up={cfg.rollout_n_context}, rollout={cfg.rollout_n_rollout})",
    )
    figs["coherence_per_component"] = plot_per_component_bars(
        comp_labels,
        {"Linear probe": c3.coherence_metrics.per_component_scores},
        title="Per-component trajectory coherence  (lower = smoother)",
    )
    figs["coherence_distribution"] = plot_coherence_distribution(
        {"Linear probe": c3.per_sample_scores},
        title="Trajectory coherence score distribution  (AR rollout)",
    )

    with h5py.File(cfg.test_h5_path, "r") as f:
        colors_all = f["colors"][:3, :cfg.n_obj]

    for i in range(min(3, c3.decoded_pos_roll.shape[0])):
        gt_slice = s.positions_gt[
            i,
            cfg.rollout_n_context : cfg.rollout_n_context + cfg.rollout_n_rollout,
            :cfg.n_obj,
        ]
        fig = plot_trajectory_comparison(
            gt_slice,
            {"Linear probe (AR)": c3.decoded_pos_roll[i]},
            scene_colors=colors_all[i],
            title=f"Sample {i}  —  decoded positions during AR rollout",
        )
        figs[f"rollout_traj_{i}"] = fig

    return figs


def plot_criterion4(cfg: EvalConfig, s: SetupResult, c2: C2Result, c4: C4Result) -> dict[str, Figure]:
    from pim.editors.probe_steering import inject_state, probe_decomposition

    figs: dict[str, Figure] = {}
    A, b, A_pinv = probe_decomposition(c2.linear_extractor)

    for i in range(min(3, c4.h_at_edit.shape[0])):
        h      = torch.from_numpy(c4.h_at_edit[i]).float().to(cfg.device)
        target = torch.from_numpy(c4.env_state_targets[i]).float().to(cfg.device)
        h_edited = inject_state(h.unsqueeze(0), target.unsqueeze(0), A, A_pinv, b)

        obs_start   = torch.from_numpy(c4.obs_at_edit[i]).float().to(cfg.device).unsqueeze(0)
        h_gru_s     = h_edited.unsqueeze(0)
        h_gru_u     = h.unsqueeze(0).unsqueeze(0)
        x_s, x_u   = obs_start, obs_start
        steered_preds, unsteered_preds = [], []

        with torch.no_grad():
            for _ in range(cfg.ctrl_n_rollout):
                x_s, h_gru_s = s.model.step(x_s, h_gru_s)
                x_u, h_gru_u = s.model.step(x_u, h_gru_u)
                steered_preds.append(x_s.squeeze(0).cpu().numpy())
                unsteered_preds.append(x_u.squeeze(0).cpu().numpy())

        steered_obs   = np.stack(steered_preds)
        unsteered_obs = np.stack(unsteered_preds)
        gt_post       = c4.obs_post_edit[i, :cfg.ctrl_n_rollout]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#ffffff")
        for ax, img, ttl in zip(
            axes,
            [gt_post, steered_obs, unsteered_obs],
            ["GT (post-edit)", "Steered rollout", "Unsteered rollout"],
        ):
            style_ax(ax)
            ax.imshow(np.clip(img, 0, 1), aspect="auto", origin="upper",
                      interpolation="nearest", cmap="gray", vmin=0, vmax=1)
            ax.set_title(ttl, fontsize=10)
            ax.set_xlabel("ray position", fontsize=9)
            ax.set_ylabel("frame", fontsize=9)
        fig.suptitle(f"Sample {i}  —  counterfactual controllability", fontsize=11)
        plt.tight_layout()
        figs[f"controllability_{i}"] = fig

    return figs


# ── Persistence helpers ───────────────────────────────────────────────────────


def save_figures(figures: dict[str, Figure], output_dir: Path, dpi: int = 150) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(figures)} figures → {output_dir}")


def save_metrics(results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {}
    if "c1" in results:
        c1 = results["c1"]
        metrics["criterion1"] = {
            "next_step_mse_mean": c1.metrics.mean_mse,
            "next_step_mse_std":  c1.metrics.std_mse,
        }
    if "c2" in results:
        c2 = results["c2"]
        metrics["criterion2"] = {
            "linear_overall_mse": c2.recovery_linear.overall_mse,
            "mlp_overall_mse":    c2.recovery_mlp.overall_mse,
        }
    if "c3" in results:
        c3 = results["c3"]
        metrics["criterion3"] = {
            "coherence_mean":   c3.coherence_metrics.mean_score,
            "coherence_std":    c3.coherence_metrics.std_score,
            "mean_jump_ratio":  c3.coherence_metrics.mean_jump_ratio,
        }
    if "c4" in results:
        c4 = results["c4"]
        metrics["criterion4"] = {
            "steered_mse":    c4.ctrl_metrics.steered_mse,
            "unsteered_mse":  c4.ctrl_metrics.unsteered_mse,
            "injection_error": c4.ctrl_metrics.injection_error,
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  metrics → {output_dir / 'metrics.json'}")


def save_config(cfg: EvalConfig, s: SetupResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "eval_config": cfg.__dict__,
        "checkpoint": {
            "epoch":        s.ckpt_info["epoch"],
            "val_loss":     s.ckpt_info["val_loss"],
            "model_config": s.ckpt_info["model_config"],
        },
    }
    (output_dir / "eval_config.json").write_text(json.dumps(out, indent=2))


# ── Convenience runner ────────────────────────────────────────────────────────


def run_all(cfg: EvalConfig) -> dict:
    """Run setup + all requested criteria. Returns dict with keys setup/c1/c2/c3/c4."""
    s = setup(cfg)
    results: dict = {"setup": s}

    if 1 in cfg.criteria:
        results["c1"] = run_criterion1(cfg, s)
    if 2 in cfg.criteria:
        if "c1" not in results:
            raise ValueError("Criterion 2 requires criterion 1 (needs internal_states_tf)")
        results["c2"] = run_criterion2(cfg, s, results["c1"])
    if 3 in cfg.criteria:
        if "c2" not in results:
            raise ValueError("Criterion 3 requires criterion 2 (needs linear_extractor)")
        results["c3"] = run_criterion3(cfg, s, results["c2"])
    if 4 in cfg.criteria:
        if "c2" not in results:
            raise ValueError("Criterion 4 requires criterion 2 (needs linear_extractor)")
        results["c4"] = run_criterion4(cfg, s, results["c2"])

    return results
