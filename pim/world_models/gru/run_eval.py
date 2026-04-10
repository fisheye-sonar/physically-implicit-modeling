"""End-to-end GRU evaluation pipeline.

Implements all four evaluation criteria as pure functions over typed result
dataclasses. Designed for two calling contexts:

  Notebook (interactive):
      from pim.world_models.gru.run_eval import EvalConfig, setup, run_criterion1, plot_criterion1
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
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pim.eval._helpers import collect_rollout, run_autoregressive, run_teacher_forcing
from pim.eval.controllability import ControllabilityMetrics
from pim.eval.plotting import (
    PALETTE,
    _BG_HEX,
    _TEXT_COLOR,
    _TICK_COLOR,
    plot_color,
    plot_coherence_distribution,
    plot_per_component_bars,
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
from pim.simulator.dataset import load_sample
from pim.simulator.viz import (
    _BG as _DARK_BG_ARRAY,
    _BG_HEX as _DARK_BG_HEX,
    _TEXT_COLOR as _DARK_TEXT_COLOR,
    _TICK_COLOR as _DARK_TICK_COLOR,
    make_waterfall,
)
from pim.world_models.dataloader import ObservationDataset
from pim.world_models.gru import GRUModel, ModelConfig


# ── Dark theme helper (simulator aesthetic for waterfall panels) ──────────────


def _style_ax_dark(ax) -> None:
    ax.set_facecolor(_DARK_BG_HEX)
    for spine in ax.spines.values():
        spine.set_edgecolor(_DARK_TICK_COLOR)
    ax.tick_params(colors=_DARK_TICK_COLOR, labelsize=9)


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
    probe_hidden_dim: int = 256  # MLP hidden layer width

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
    obs_actual: np.ndarray  # (N, T, R)
    positions_gt: np.ndarray  # (N, T, n_obj, 2)
    is_visible: np.ndarray  # (N, T, n_obj)
    run_name: str
    T_frames: int
    obs_res: int
    metrics_history: list[dict]


@dataclass
class C1Result:
    """Criterion 1 — Predictive Quality."""

    metrics: PredictionMetrics
    obs_pred_tf: np.ndarray  # (N, T-1, R)  teacher-forcing predictions
    internal_states_tf: np.ndarray  # (N, T-1, H)  hidden states
    obs_rollout: np.ndarray  # (N, T-n_context, R)  AR predictions
    horizon_mse: np.ndarray  # (T-n_context,)  MSE at each horizon step
    context_lengths: np.ndarray  # (T-1,)
    mse_by_ctx: np.ndarray  # (T-1,)  1-step MSE as function of context


@dataclass
class C2Result:
    """Criterion 2 — Recovery."""

    linear_extractor: LinearExtractor
    mlp_extractor: MLPExtractor
    recovery_linear: RecoveryMetrics
    recovery_mlp: RecoveryMetrics
    env_states_tf: np.ndarray  # (N, T-1, n_obj, 2)  GT states aligned to h_tf
    vis_mask_tf: np.ndarray  # (N, T-1)  both-visible mask


@dataclass
class C3Result:
    """Criterion 3 — Rollout Consistency."""

    drift_mse: np.ndarray  # (n_rollout,)
    coherence_metrics: CoherenceMetrics
    mlp_coherence_metrics: CoherenceMetrics
    decoded_pos_roll: np.ndarray  # (N_eval, n_rollout, n_obj, 2) — linear
    mlp_decoded_pos_roll: np.ndarray  # (N_eval, n_rollout, n_obj, 2) — MLP
    per_sample_scores: np.ndarray  # (N_eval,) — linear coherence scores
    mlp_per_sample_scores: np.ndarray  # (N_eval,) — MLP coherence scores
    gt_per_sample_scores: np.ndarray  # (N_eval,) — GT coherence scores
    obs_rollout_co: np.ndarray  # (N_eval, n_rollout, R) — predicted observations


@dataclass
class C4Result:
    """Criterion 4 — Counterfactual Controllability."""

    ctrl_metrics: ControllabilityMetrics
    h_at_edit: np.ndarray  # (N, H)
    env_state_targets: np.ndarray  # (N, n_obj*2)
    obs_at_edit: np.ndarray  # (N, R)
    obs_post_edit: np.ndarray  # (N, T-edit_frame, R)
    edit_frame: int
    # Per-step MSE curves (averaged over N samples)
    steered_obs_step: np.ndarray  # (ctrl_n_rollout,)
    unsteered_obs_step: np.ndarray  # (ctrl_n_rollout,)
    steered_pos_step: np.ndarray  # (ctrl_n_rollout,)
    unsteered_pos_step: np.ndarray  # (ctrl_n_rollout,)
    # First few samples for waterfall + position viz
    viz_steered: np.ndarray  # (n_viz, ctrl_n_rollout, R)
    viz_unsteered: np.ndarray  # (n_viz, ctrl_n_rollout, R)
    viz_obs_pre_edit: np.ndarray  # (n_viz, edit_frame, R) — actual pre-edit obs
    # Steered: n_rollout+1 points — index 0 = decoded h' (aligns with GT at edit_frame)
    viz_steered_pos: np.ndarray  # (n_viz, ctrl_n_rollout+1, n_obj, 2)
    # Unsteered: n_rollout points at frames edit_frame+1..edit_frame+n_rollout
    viz_unsteered_pos: np.ndarray  # (n_viz, ctrl_n_rollout, n_obj, 2)
    # GT: n_rollout+1 frames starting at edit_frame (same x-axis as viz_steered_pos)
    viz_gt_pos: np.ndarray  # (n_viz, ctrl_n_rollout+1, n_obj, 2)
    # Pre-edit context: last min(8, edit_frame) frames before edit
    viz_pre_edit_pos: np.ndarray  # (n_viz, n_ctx_show, n_obj, 2)
    viz_colors: np.ndarray  # (n_viz, n_obj, 3)


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
        "epoch": ckpt["epoch"],
        "val_loss": ckpt["val_loss"],
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
        T_frames = f["obs_intensity"].shape[1]
        obs_res = f["obs_intensity"].shape[2]
        obs_actual = f["obs_intensity"][:].astype(np.float32)
        positions_gt = f["positions"][:, :, : cfg.n_obj, :].astype(np.float32)
        is_visible = f["is_visible"][:, :, : cfg.n_obj].astype(bool)
        n_samples = f["obs_intensity"].shape[0]

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
        s.model,
        s.test_loader,
        cfg.device,
    )
    metrics = eval_single_step(s.obs_actual, obs_pred_tf)
    print(f"  next-step MSE: {metrics.mean_mse:.6f} ± {metrics.std_mse:.6f}")

    print("  AR rollout (all test samples)...")
    all_rollouts = []
    for i in tqdm(range(len(s.obs_actual)), desc="  AR rollout", leave=False):
        rollout, _ = run_autoregressive(
            s.model, s.obs_actual[i], cfg.n_context, cfg.device
        )
        all_rollouts.append(rollout)
    obs_rollout = np.stack(all_rollouts)
    horizon_mse = eval_horizon_mse(s.obs_actual, obs_rollout, cfg.n_context)
    print(
        f"  horizon MSE  step 1={horizon_mse[0]:.4f}  step {len(horizon_mse)}={horizon_mse[-1]:.4f}"
    )

    print("  MSE by context length...")
    context_lengths, mse_by_ctx = eval_mse_by_context(
        s.model,
        s.test_loader,
        n_steps_ahead=1,
        device=cfg.device,
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
    env_states_tf = s.positions_gt[:, :-1, : cfg.n_obj, :]  # (N, T-1, n_obj, 2)
    vis_mask_tf = s.is_visible[:, :-1, : cfg.n_obj].all(axis=2)  # (N, T-1)
    loss_fn = hungarian_mse if cfg.use_hungarian else identity_mse

    print("  training LinearExtractor...")
    linear_extractor = LinearExtractor(s.model.hidden_size, state_def)
    if cfg.use_lstsq:
        mse = fit_lstsq(
            linear_extractor,
            c1.internal_states_tf,
            env_states_tf,
            mask=vis_mask_tf,
        )
        print(f"  lstsq train MSE: {mse:.6f}")
    else:
        losses = train_extractor(
            linear_extractor,
            c1.internal_states_tf,
            env_states_tf,
            n_epochs=cfg.probe_n_epochs,
            lr=cfg.probe_lr,
            loss_fn=loss_fn,
            mask=vis_mask_tf,
            device=cfg.device,
        )
        print(f"  final train loss: {losses[-1]:.6f}")

    print("  training MLPExtractor...")
    mlp_extractor = MLPExtractor(
        s.model.hidden_size, state_def, mlp_hidden=cfg.probe_hidden_dim
    )
    mlp_losses = train_extractor(
        mlp_extractor,
        c1.internal_states_tf,
        env_states_tf,
        n_epochs=cfg.probe_n_epochs,
        lr=cfg.probe_lr,
        loss_fn=loss_fn,
        mask=vis_mask_tf,
        device=cfg.device,
    )
    print(f"  MLP final train loss: {mlp_losses[-1]:.6f}")

    recovery_linear = eval_recovery(
        env_states_tf,
        c1.internal_states_tf,
        linear_extractor,
        mask=vis_mask_tf,
        use_hungarian=cfg.use_hungarian,
        device=cfg.device,
    )
    recovery_mlp = eval_recovery(
        env_states_tf,
        c1.internal_states_tf,
        mlp_extractor,
        mask=vis_mask_tf,
        use_hungarian=cfg.use_hungarian,
        device=cfg.device,
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
            s.model,
            s.obs_actual[i],
            cfg.rollout_n_context,
            cfg.rollout_n_rollout,
            cfg.device,
        )
        all_h_roll.append(h_roll)
        all_obs_roll.append(obs_roll)

    h_rollout_arr = np.stack(all_h_roll)  # (N_eval, n_rollout, H)
    obs_rollout_co = np.stack(all_obs_roll)  # (N_eval, n_rollout, R)

    drift_mse = eval_observation_drift(
        s.obs_actual[: cfg.coherence_n_eval],
        obs_rollout_co,
        cfg.rollout_n_context,
    )

    # Decode positions with both extractors
    c2.linear_extractor.eval()
    c2.mlp_extractor.eval()
    with torch.no_grad():
        h_t = torch.from_numpy(h_rollout_arr).float().to(cfg.device)
        decoded_pos_roll = c2.linear_extractor(h_t).cpu().numpy()
        mlp_decoded_pos_roll = c2.mlp_extractor(h_t).cpu().numpy()

    coherence_metrics = eval_trajectory_coherence(decoded_pos_roll)
    mlp_coherence_metrics = eval_trajectory_coherence(mlp_decoded_pos_roll)

    per_sample_scores = np.array(
        [rollout_coherence(decoded_pos_roll[i])[0] for i in range(cfg.coherence_n_eval)]
    )
    mlp_per_sample_scores = np.array(
        [rollout_coherence(mlp_decoded_pos_roll[i])[0] for i in range(cfg.coherence_n_eval)]
    )

    # GT coherence scores over the rollout window
    n_ctx = cfg.rollout_n_context
    n_roll = cfg.rollout_n_rollout
    gt_per_sample_scores = np.array(
        [
            rollout_coherence(s.positions_gt[i, n_ctx : n_ctx + n_roll, : cfg.n_obj])[0]
            for i in range(cfg.coherence_n_eval)
        ]
    )

    print(
        f"  linear coherence  mean={coherence_metrics.mean_score:.4f}  "
        f"jump ratio={coherence_metrics.mean_jump_ratio:.2f}"
    )
    print(
        f"  MLP    coherence  mean={mlp_coherence_metrics.mean_score:.4f}  "
        f"jump ratio={mlp_coherence_metrics.mean_jump_ratio:.2f}"
    )
    print(f"  GT     coherence  mean={gt_per_sample_scores.mean():.4f}")

    return C3Result(
        drift_mse=drift_mse,
        coherence_metrics=coherence_metrics,
        mlp_coherence_metrics=mlp_coherence_metrics,
        decoded_pos_roll=decoded_pos_roll,
        mlp_decoded_pos_roll=mlp_decoded_pos_roll,
        per_sample_scores=per_sample_scores,
        mlp_per_sample_scores=mlp_per_sample_scores,
        gt_per_sample_scores=gt_per_sample_scores,
        obs_rollout_co=obs_rollout_co,
    )


# ── Criterion 4: Controllability ──────────────────────────────────────────────


def run_criterion4(cfg: EvalConfig, s: SetupResult, c2: C2Result) -> C4Result:
    print("Criterion 4 — Counterfactual Controllability")

    with h5py.File(cfg.edits_h5_path, "r") as f:
        edit_frames = f["edit_frame"][:].astype(int)
        edits_obs = f["obs_intensity"][:].astype(np.float32)
        edits_positions = f["positions"][:].astype(np.float32)
        edits_colors = f["colors"][:].astype(np.float32)

    edit_frame = int(edit_frames[0])
    n_ctrl = min(500, edits_obs.shape[0])
    n_viz = min(3, n_ctrl)
    print(f"  edit_frame={edit_frame}  n_ctrl={n_ctrl}")

    # Warm up model to edit_frame for each sample (teacher-force on actual obs)
    h_at_edit = np.zeros((n_ctrl, s.model.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(n_ctrl), desc="  teacher-force to edit frame", leave=False):
            obs_seq = torch.from_numpy(edits_obs[i]).float().to(cfg.device)
            h = None
            for t in range(edit_frame):
                _, h = s.model.step(obs_seq[t].unsqueeze(0), h)
            h_at_edit[i] = h[0, 0].cpu().numpy()

    env_state_targets = edits_positions[:n_ctrl, edit_frame, : cfg.n_obj, :].reshape(
        n_ctrl, cfg.n_obj * 2
    )
    obs_at_edit = edits_obs[:n_ctrl, edit_frame, :]
    obs_post_edit = edits_obs[:n_ctrl, edit_frame:, :]

    # Inject edit and collect per-step rollout data
    from pim.editors.probe_steering import inject_state, probe_decomposition

    c2.linear_extractor.to(cfg.device).eval()
    A, b, A_pinv = probe_decomposition(c2.linear_extractor)

    steered_obs_all: list[np.ndarray] = []
    unsteered_obs_all: list[np.ndarray] = []
    steered_h_all: list[np.ndarray] = []
    unsteered_h_all: list[np.ndarray] = []
    inj_errs: list[float] = []

    with torch.no_grad():
        for i in tqdm(range(n_ctrl), desc="  controllability rollouts", leave=False):
            h = torch.from_numpy(h_at_edit[i]).float().to(cfg.device)
            target = torch.from_numpy(env_state_targets[i]).float().to(cfg.device)
            h_edited = inject_state(h.unsqueeze(0), target.unsqueeze(0), A, A_pinv, b)

            readback = (h_edited @ A.T + b).squeeze(0)
            inj_errs.append(float(((readback - target) ** 2).mean().item()))

            obs_start = (
                torch.from_numpy(obs_at_edit[i]).float().to(cfg.device).unsqueeze(0)
            )

            # Steered rollout — h' is index 0 so decode(h') == target exactly
            h_gru_s = h_edited.unsqueeze(0)
            x_s = obs_start
            s_obs = []
            s_h = [h_edited.squeeze(0).cpu().numpy()]  # h' at index 0
            for _ in range(cfg.ctrl_n_rollout):
                x_s, h_gru_s = s.model.step(x_s, h_gru_s)
                s_obs.append(x_s.squeeze(0).cpu().numpy())
                s_h.append(h_gru_s[-1, 0].cpu().numpy())  # post-step states at indices 1..
            steered_obs_all.append(np.stack(s_obs))
            steered_h_all.append(np.stack(s_h))  # length n_rollout+1

            # Unsteered rollout
            h_gru_u = h.unsqueeze(0).unsqueeze(0)
            x_u = obs_start
            u_obs, u_h = [], []
            for _ in range(cfg.ctrl_n_rollout):
                x_u, h_gru_u = s.model.step(x_u, h_gru_u)
                u_obs.append(x_u.squeeze(0).cpu().numpy())
                u_h.append(h_gru_u[-1, 0].cpu().numpy())
            unsteered_obs_all.append(np.stack(u_obs))
            unsteered_h_all.append(np.stack(u_h))

    steered_obs_arr = np.stack(steered_obs_all)    # (N, ctrl_n_rollout, R)
    unsteered_obs_arr = np.stack(unsteered_obs_all)
    # steered_h_arr[i, 0] = h' (injected), [i, 1..] = post-step hidden states
    steered_h_arr = np.stack(steered_h_all)        # (N, ctrl_n_rollout+1, H)
    unsteered_h_arr = np.stack(unsteered_h_all)    # (N, ctrl_n_rollout, H)

    gt_obs = obs_post_edit[:n_ctrl, : cfg.ctrl_n_rollout]  # (N, ctrl_n_rollout, R)
    steered_obs_step = ((steered_obs_arr - gt_obs) ** 2).mean(axis=(0, 2))
    unsteered_obs_step = ((unsteered_obs_arr - gt_obs) ** 2).mean(axis=(0, 2))

    # Per-step position MSE: compare post-step states (indices 1..) vs GT at ef+1..ef+n_rollout
    gt_positions_mse = edits_positions[
        :n_ctrl, edit_frame + 1 : edit_frame + cfg.ctrl_n_rollout + 1, : cfg.n_obj, :
    ]  # (N, ctrl_n_rollout, n_obj, 2)
    with torch.no_grad():
        # Post-step steered states: indices 1.. of steered_h_arr
        h_s_post = torch.from_numpy(steered_h_arr[:, 1:, :]).float().to(cfg.device)
        h_u_t = torch.from_numpy(unsteered_h_arr).float().to(cfg.device)
        steered_pos_post = c2.linear_extractor(h_s_post).cpu().numpy()  # (N, n_rollout, n_obj, 2)
        unsteered_pos = c2.linear_extractor(h_u_t).cpu().numpy()

    steered_pos_step = ((steered_pos_post - gt_positions_mse) ** 2).mean(axis=(0, 2, 3))
    unsteered_pos_step = ((unsteered_pos - gt_positions_mse) ** 2).mean(axis=(0, 2, 3))

    # Decode full steered sequence (n_rollout+1) for viz: index 0 aligns with edit_frame GT
    with torch.no_grad():
        h_s_all = torch.from_numpy(steered_h_arr[:n_viz]).float().to(cfg.device)
        h_u_viz = torch.from_numpy(unsteered_h_arr[:n_viz]).float().to(cfg.device)
        viz_steered_pos = c2.linear_extractor(h_s_all).cpu().numpy()   # (n_viz, n_rollout+1, n_obj, 2)
        viz_unsteered_pos = c2.linear_extractor(h_u_viz).cpu().numpy() # (n_viz, n_rollout, n_obj, 2)

    # GT positions for viz: n_rollout+1 frames starting at edit_frame
    viz_gt_pos = edits_positions[
        :n_viz, edit_frame : edit_frame + cfg.ctrl_n_rollout + 1, : cfg.n_obj, :
    ]

    n_ctx_show = min(8, edit_frame)
    viz_pre_edit_pos = edits_positions[
        :n_viz, edit_frame - n_ctx_show : edit_frame, : cfg.n_obj, :
    ]

    ctrl_metrics = ControllabilityMetrics(
        steered_mse=float(((steered_obs_arr - gt_obs) ** 2).mean()),
        unsteered_mse=float(((unsteered_obs_arr - gt_obs) ** 2).mean()),
        injection_error=float(np.mean(inj_errs)),
    )
    ratio = ctrl_metrics.unsteered_mse / (ctrl_metrics.steered_mse + 1e-12)
    print(
        f"  steered={ctrl_metrics.steered_mse:.6f}  "
        f"unsteered={ctrl_metrics.unsteered_mse:.6f}  ratio={ratio:.2f}x"
    )

    return C4Result(
        ctrl_metrics=ctrl_metrics,
        h_at_edit=h_at_edit,
        env_state_targets=env_state_targets,
        obs_at_edit=obs_at_edit,
        obs_post_edit=obs_post_edit,
        edit_frame=edit_frame,
        steered_obs_step=steered_obs_step,
        unsteered_obs_step=unsteered_obs_step,
        steered_pos_step=steered_pos_step,
        unsteered_pos_step=unsteered_pos_step,
        viz_steered=steered_obs_arr[:n_viz],
        viz_unsteered=unsteered_obs_arr[:n_viz],
        viz_obs_pre_edit=edits_obs[:n_viz, :edit_frame],
        viz_steered_pos=viz_steered_pos,
        viz_unsteered_pos=viz_unsteered_pos,
        viz_gt_pos=viz_gt_pos,
        viz_pre_edit_pos=viz_pre_edit_pos,
        viz_colors=edits_colors[:n_viz, : cfg.n_obj],
    )


# ── Plot functions — return dict[str, Figure] ─────────────────────────────────


def plot_setup(cfg: EvalConfig, s: SetupResult, n_dataset_show: int = 8) -> dict[str, Figure]:
    """Training curves + dataset overview grid."""
    figs: dict[str, Figure] = {}

    epochs = [m["epoch"] for m in s.metrics_history]
    train_loss = [m["train_loss"] for m in s.metrics_history]
    val_loss = [m["val_loss"] for m in s.metrics_history]
    best_epoch = s.ckpt_info["epoch"]

    for log_x, key in [(False, "training_linlog"), (True, "training_loglog")]:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
        style_ax(ax)
        ax.plot(epochs, train_loss, color=PALETTE[0], linewidth=1.8, label="train")
        ax.plot(
            epochs, val_loss, color=PALETTE[1], linewidth=1.8, label="val",
            linestyle="--",
        )
        ax.axvline(
            best_epoch, color=_TICK_COLOR, linewidth=1.0, linestyle=":",
            alpha=0.7, label=f"best epoch {best_epoch}",
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch", color=_TEXT_COLOR, fontsize=10)
        ax.set_ylabel("loss", color=_TEXT_COLOR, fontsize=10)
        ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
        fig.suptitle(
            s.run_name, color=_TEXT_COLOR, fontsize=12, fontweight="bold", y=0.99,
        )
        ax.set_title(
            "training curves" + (" (log-log)" if log_x else ""),
            color=_TEXT_COLOR, fontsize=10,
        )
        plt.tight_layout()
        figs[key] = fig

    # Dataset overview — dark simulator aesthetic
    n_cols = n_dataset_show // 2
    fig_ds, axes = plt.subplots(
        2, n_cols,
        figsize=(n_cols * 1.8, 6),
        facecolor=_DARK_BG_HEX,
    )
    fig_ds.suptitle(
        "dataset overview — stored observations",
        color=_DARK_TEXT_COLOR, fontsize=12,
    )
    for ax, idx in zip(axes.flat, range(n_dataset_show)):
        scene_i, obs_depth_i, obs_id_i, obs_intensity_i = load_sample(
            cfg.test_h5_path, idx
        )
        wf = make_waterfall(obs_depth_i, obs_id_i, obs_intensity_i, scene_i)
        _style_ax_dark(ax)
        ax.imshow(wf, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_title(f"#{idx}", color=_DARK_TEXT_COLOR, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    figs["dataset_overview"] = fig_ds

    return figs


def plot_criterion1(cfg: EvalConfig, s: SetupResult, c1: C1Result) -> dict[str, Figure]:
    """MSE vs context length only. Horizon MSE is shown in Criterion 3."""
    figs: dict[str, Figure] = {}

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(c1.context_lengths, c1.mse_by_ctx, color=PALETTE[0], linewidth=1.8)
    ax.set_xlabel("context frames", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"1-step prediction MSE vs context length  (warm-up={cfg.n_context})",
        color=_TEXT_COLOR, fontsize=11,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    figs["mse_by_context"] = fig

    return figs


def plot_criterion2(
    cfg: EvalConfig,
    s: SetupResult,
    c1: C1Result,
    c2: C2Result,
    n_traj: int = 3,
) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}

    # Per-object MSE bar
    lin_per_obj = c2.recovery_linear.per_component_mse.reshape(cfg.n_obj, 2).mean(1)
    mlp_per_obj = c2.recovery_mlp.per_component_mse.reshape(cfg.n_obj, 2).mean(1)
    obj_labels = [f"obj {i}" for i in range(cfg.n_obj)]
    figs["recovery_per_object"] = plot_per_component_bars(
        obj_labels,
        {"linear": lin_per_obj, "MLP": mlp_per_obj},
        title="Per-object position recovery MSE",
    )

    # Recovery MSE vs context length
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(
        np.arange(1, len(c2.recovery_linear.mse_by_context) + 1),
        c2.recovery_linear.mse_by_context,
        color=PALETTE[0], linewidth=1.8, label="linear",
    )
    ax.plot(
        np.arange(1, len(c2.recovery_mlp.mse_by_context) + 1),
        c2.recovery_mlp.mse_by_context,
        color=PALETTE[1], linewidth=1.8, label="MLP", linestyle="--",
    )
    ax.set_xlabel("context frames", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("position recovery MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title("Position recovery MSE vs context length", color=_TEXT_COLOR, fontsize=11)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    plt.tight_layout()
    figs["recovery_by_context"] = fig

    # Trajectory viz: x and y line plots — GT solid, linear ×, MLP ○
    with h5py.File(cfg.test_h5_path, "r") as f:
        colors_all = f["colors"][:n_traj, : cfg.n_obj]

    c2.linear_extractor.eval()
    c2.mlp_extractor.eval()
    for i in range(n_traj):
        with torch.no_grad():
            h_i = (
                torch.from_numpy(c1.internal_states_tf[i : i + 1])
                .float()
                .to(cfg.device)
            )
            lin_pos = c2.linear_extractor(h_i).cpu().numpy()[0]  # (T-1, n_obj, 2)
            mlp_pos = c2.mlp_extractor(h_i).cpu().numpy()[0]

        gt_pos = s.positions_gt[i, :-1, : cfg.n_obj]  # (T-1, n_obj, 2)
        vis = c2.vis_mask_tf[i]  # (T-1,)
        timesteps = np.arange(gt_pos.shape[0])
        vis_t = timesteps[vis]

        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
        fig.suptitle(
            f"Sample {i}  —  position recovery (teacher forcing)",
            color=_TEXT_COLOR, fontsize=11, fontweight="bold",
        )
        for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
            style_ax(ax)
            for obj in range(cfg.n_obj):
                color = plot_color(colors_all[i, obj])
                ax.plot(
                    timesteps, gt_pos[:, obj, coord],
                    color=color, linewidth=1.8,
                )
                ax.scatter(
                    vis_t, lin_pos[vis, obj, coord],
                    color=color, s=18, marker="x", alpha=0.75,
                )
                ax.scatter(
                    vis_t, mlp_pos[vis, obj, coord],
                    color=color, s=18, marker="o", alpha=0.75,
                )
            ax.set_xlabel("frame", color=_TEXT_COLOR, fontsize=9)
            ax.set_ylabel(coord_lbl, color=_TEXT_COLOR, fontsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(colors=_TICK_COLOR)

        handles = [
            Line2D([0], [0], color="gray", linewidth=1.8, label="GT"),
            Line2D(
                [0], [0], color="gray", marker="x", linestyle="none",
                markersize=6, label="linear",
            ),
            Line2D(
                [0], [0], color="gray", marker="o", linestyle="none",
                markersize=6, label="MLP",
            ),
        ]
        axes[0].legend(handles=handles, frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
        plt.tight_layout()
        figs[f"recovery_traj_{i}"] = fig

    return figs


def plot_criterion3(
    cfg: EvalConfig,
    s: SetupResult,
    c1: C1Result,
    c3: C3Result,
    n_viz: int = 3,
) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}

    # Observation drift (= horizon MSE from C1, displayed here)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(
        np.arange(1, len(c1.horizon_mse) + 1), c1.horizon_mse,
        color=PALETTE[0], linewidth=1.8,
    )
    ax.set_xlabel("steps ahead", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        f"Observation drift  (warm-up={cfg.rollout_n_context}, rollout={cfg.rollout_n_rollout})",
        color=_TEXT_COLOR, fontsize=11,
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    figs["observation_drift"] = fig

    # Coherence bar chart: GT vs linear vs MLP
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_BG_HEX)
    style_ax(ax)
    names = ["GT", "linear", "MLP"]
    means = [
        c3.gt_per_sample_scores.mean(),
        c3.per_sample_scores.mean(),
        c3.mlp_per_sample_scores.mean(),
    ]
    ax.bar(names, means, color=[PALETTE[3], PALETTE[0], PALETTE[1]], width=0.5)
    ax.set_ylabel("mean coherence score", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        "Trajectory coherence (lower = smoother)", color=_TEXT_COLOR, fontsize=11,
    )
    ax.tick_params(colors=_TICK_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_TICK_COLOR)
    plt.tight_layout()
    figs["coherence_bar"] = fig

    # Coherence distribution — GT + linear + MLP
    figs["coherence_distribution"] = plot_coherence_distribution(
        {
            "GT": c3.gt_per_sample_scores,
            "linear": c3.per_sample_scores,
            "MLP": c3.mlp_per_sample_scores,
        },
        title="Trajectory coherence score distribution  (AR rollout)",
    )

    n_ctx = cfg.rollout_n_context
    n_roll = cfg.rollout_n_rollout
    _n_ctx_show = min(8, n_ctx)

    for i in range(min(n_viz, cfg.coherence_n_eval)):
        scene_i, obs_depth_i, obs_id_i, obs_intensity_i = load_sample(
            cfg.test_h5_path, i
        )
        colors_i = scene_i.colors

        pos_v = s.positions_gt[i, :, : cfg.n_obj]  # (T, n_obj, 2)
        lin_roll_v = c3.decoded_pos_roll[i]          # (n_roll, n_obj, 2)
        mlp_roll_v = c3.mlp_decoded_pos_roll[i]

        s_gt = c3.gt_per_sample_scores[i]
        s_lin = c3.per_sample_scores[i]
        s_mlp = c3.mlp_per_sample_scores[i]

        _ctx_frames = np.arange(n_ctx - _n_ctx_show, n_ctx)
        _roll_frames = np.arange(n_ctx, n_ctx + n_roll)

        # ── x/y time plots ────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
        fig.suptitle(
            f"Sample {i}  —  AR rollout positions  "
            f"coherence: GT={s_gt:.3f}  linear={s_lin:.3f}  MLP={s_mlp:.3f}",
            color=_TEXT_COLOR, fontsize=11, fontweight="bold",
        )
        for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
            style_ax(ax)
            for obj in range(cfg.n_obj):
                color = plot_color(colors_i[obj])
                ax.plot(
                    _ctx_frames, pos_v[n_ctx - _n_ctx_show : n_ctx, obj, coord],
                    color=color, linewidth=1.5, alpha=0.3,
                )
                ax.plot(
                    _roll_frames, pos_v[n_ctx : n_ctx + n_roll, obj, coord],
                    color=color, linewidth=1.8, alpha=0.9,
                )
                ax.scatter(
                    _roll_frames, lin_roll_v[:, obj, coord],
                    color=color, s=20, marker="x", alpha=0.9,
                )
                ax.scatter(
                    _roll_frames, mlp_roll_v[:, obj, coord],
                    color=color, s=20, marker="o", alpha=0.9,
                )
            ax.axvline(
                n_ctx - 0.5, color=_TICK_COLOR, linewidth=1.0, linestyle="--", alpha=0.5,
            )
            ax.set_xlabel("frame", color=_TEXT_COLOR, fontsize=9)
            ax.set_ylabel(coord_lbl, color=_TEXT_COLOR, fontsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(colors=_TICK_COLOR)

        handles = [
            Line2D([0], [0], color="gray", linewidth=1.8, label="GT"),
            Line2D(
                [0], [0], color="gray", marker="x", linestyle="none",
                markersize=6, label="linear",
            ),
            Line2D(
                [0], [0], color="gray", marker="o", linestyle="none",
                markersize=6, label="MLP",
            ),
        ]
        axes[0].legend(handles=handles, frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
        plt.tight_layout()
        figs[f"rollout_traj_{i}"] = fig

        # ── 3-panel: actual waterfall | predicted waterfall | 2D positions ─
        wf_actual = make_waterfall(
            obs_depth_i, obs_id_i, obs_intensity_i, scene_i, mode="model"
        )
        obs_roll_i = c3.obs_rollout_co[i]  # (n_roll, R)

        wf_pred = np.zeros_like(wf_actual)
        wf_pred[:, :, :3] = _DARK_BG_ARRAY
        wf_pred[:, :, 3] = 1.0
        wf_pred[:n_ctx] = wf_actual[:n_ctx] * np.array([1.0, 1.0, 1.0, 0.35])
        wf_pred[:n_ctx, :, 3] = 1.0
        gray = np.clip(obs_roll_i, 0.0, 1.0)
        wf_pred[n_ctx : n_ctx + n_roll, :, 0] = gray
        wf_pred[n_ctx : n_ctx + n_roll, :, 1] = gray
        wf_pred[n_ctx : n_ctx + n_roll, :, 2] = gray

        fig_3 = plt.figure(figsize=(18, 5.5), facecolor=_DARK_BG_HEX)
        fig_3.suptitle(
            f"Sample {i}  —  rollout coherence  "
            f"GT={s_gt:.3f}  linear={s_lin:.3f}  MLP={s_mlp:.3f}",
            color=_DARK_TEXT_COLOR, fontsize=11, y=0.99,
        )
        fig_3.subplots_adjust(
            left=0.05, right=0.97, top=0.90, bottom=0.12, wspace=0.18,
        )

        ax_fa = fig_3.add_subplot(1, 3, 1)
        ax_fp = fig_3.add_subplot(1, 3, 2)
        ax_2d = fig_3.add_subplot(1, 3, 3)

        for ax, img, ttl in zip(
            [ax_fa, ax_fp],
            [wf_actual, wf_pred],
            ["actual", f"predicted  (warm-up={n_ctx} frames)"],
        ):
            _style_ax_dark(ax)
            ax.imshow(img, aspect="auto", origin="upper", interpolation="nearest")
            ax.axhline(
                n_ctx - 0.5, color="#fa8850", linewidth=1.2, linestyle="--", alpha=0.7,
            )
            ax.set_title(ttl, color=_DARK_TEXT_COLOR, fontsize=10)
            ax.set_xlabel("ray position", color=_DARK_TEXT_COLOR, fontsize=9)
            ax.set_ylabel("frame", color=_DARK_TEXT_COLOR, fontsize=9)

        _style_ax_dark(ax_2d)
        ax_2d.set_title("decoded positions (2D)", color=_DARK_TEXT_COLOR, fontsize=10)
        ax_2d.set_xlabel("x", color=_DARK_TEXT_COLOR, fontsize=9)
        ax_2d.set_ylabel("y (depth)", color=_DARK_TEXT_COLOR, fontsize=9)
        for obj in range(cfg.n_obj):
            color = colors_i[obj]
            ax_2d.plot(
                pos_v[:, obj, 0], pos_v[:, obj, 1],
                color=color, linewidth=1.0, alpha=0.25,
            )
            ax_2d.plot(
                pos_v[n_ctx : n_ctx + n_roll, obj, 0],
                pos_v[n_ctx : n_ctx + n_roll, obj, 1],
                color=color, linewidth=2.0, alpha=0.9,
            )
            ax_2d.scatter(
                lin_roll_v[:, obj, 0], lin_roll_v[:, obj, 1],
                color=color, s=18, marker="x", alpha=0.9,
            )
            ax_2d.scatter(
                mlp_roll_v[:, obj, 0], mlp_roll_v[:, obj, 1],
                color=color, s=18, marker="o", alpha=0.9,
            )

        figs[f"rollout_3panel_{i}"] = fig_3

    return figs


def plot_criterion4(
    cfg: EvalConfig, s: SetupResult, c2: C2Result, c4: C4Result
) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}

    steps = np.arange(1, cfg.ctrl_n_rollout + 1)

    # Per-step observation MSE
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(steps, c4.unsteered_obs_step, color=PALETTE[1], linewidth=1.8, label="unsteered")
    ax.plot(steps, c4.steered_obs_step, color=PALETTE[0], linewidth=1.8, label="steered")
    ax.set_xlabel("rollout step", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("observation MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        "Per-step observation MSE: steered vs unsteered", color=_TEXT_COLOR, fontsize=11,
    )
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    figs["ctrl_obs_mse"] = fig

    # Per-step position MSE
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_BG_HEX)
    style_ax(ax)
    ax.plot(steps, c4.unsteered_pos_step, color=PALETTE[1], linewidth=1.8, label="unsteered")
    ax.plot(steps, c4.steered_pos_step, color=PALETTE[0], linewidth=1.8, label="steered")
    ax.set_xlabel("rollout step", color=_TEXT_COLOR, fontsize=10)
    ax.set_ylabel("position MSE", color=_TEXT_COLOR, fontsize=10)
    ax.set_title(
        "Per-step position MSE: steered vs unsteered", color=_TEXT_COLOR, fontsize=11,
    )
    ax.legend(frameon=False, labelcolor=_TEXT_COLOR)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(colors=_TICK_COLOR)
    plt.tight_layout()
    figs["ctrl_pos_mse"] = fig

    n_viz = c4.viz_steered.shape[0]

    ef = c4.edit_frame
    n_ctx_show = c4.viz_pre_edit_pos.shape[1]  # min(8, edit_frame)
    ctx_frames = np.arange(ef - n_ctx_show, ef)          # pre-edit context frame indices
    roll_frames = np.arange(ef, ef + cfg.ctrl_n_rollout + 1)  # n_rollout+1 frames from ef

    for i in range(n_viz):
        colors_i = c4.viz_colors[i]  # (n_obj, 3)

        # ── x/y position line plots (frame-indexed, with pre-edit context) ─
        fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=_BG_HEX)
        fig.suptitle(
            f"Sample {i}  —  steered positions  (edit at frame {ef},  "
            f"step 0 = injected h', linear × should hit GT exactly)",
            color=_TEXT_COLOR, fontsize=11, fontweight="bold",
        )
        for ax, coord, coord_lbl in zip(axes, [0, 1], ["x", "y (depth)"]):
            style_ax(ax)
            for obj in range(cfg.n_obj):
                color = plot_color(colors_i[obj])
                # Pre-edit context — GT faint
                ax.plot(
                    ctx_frames, c4.viz_pre_edit_pos[i, :, obj, coord],
                    color=color, linewidth=1.5, alpha=0.3,
                )
                # Post-edit GT — solid (n_rollout+1 frames starting at ef)
                ax.plot(
                    roll_frames, c4.viz_gt_pos[i, :, obj, coord],
                    color=color, linewidth=1.8, alpha=0.9,
                )
                # Steered probe × — n_rollout+1 points; index 0 decodes h' → target exactly
                ax.scatter(
                    roll_frames, c4.viz_steered_pos[i, :, obj, coord],
                    color=color, s=20, marker="x", alpha=0.9,
                )
                # Unsteered probe ○ — n_rollout points starting at ef+1
                ax.scatter(
                    roll_frames[1:], c4.viz_unsteered_pos[i, :, obj, coord],
                    color=color, s=20, marker="o", alpha=0.75, facecolors="none",
                )
            ax.axvline(ef - 0.5, color=_TICK_COLOR, linewidth=1.0, linestyle="--", alpha=0.5)
            ax.set_xlabel("frame", color=_TEXT_COLOR, fontsize=9)
            ax.set_ylabel(coord_lbl, color=_TEXT_COLOR, fontsize=9)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.tick_params(colors=_TICK_COLOR)

        handles = [
            Line2D([0], [0], color="gray", linewidth=1.8, label="GT"),
            Line2D(
                [0], [0], color="gray", marker="x", linestyle="none",
                markersize=6, label="steered (h' injected)",
            ),
            Line2D(
                [0], [0], color="gray", marker="o", linestyle="none",
                markersize=6, markerfacecolor="none", label="unsteered",
            ),
        ]
        axes[0].legend(handles=handles, frameon=False, labelcolor=_TEXT_COLOR, fontsize=8)
        plt.tight_layout()
        figs[f"ctrl_pos_traj_{i}"] = fig

        # ── 3-panel dark waterfall: full sequence (pre-edit + post-edit) ──
        total_frames = c4.edit_frame + cfg.ctrl_n_rollout
        R = c4.viz_obs_pre_edit.shape[2]

        # GT: full actual sequence
        gt_obs_full = np.clip(
            np.concatenate(
                [c4.viz_obs_pre_edit[i], c4.obs_post_edit[i, : cfg.ctrl_n_rollout]],
                axis=0,
            ),
            0, 1,
        )

        # Steered/unsteered: pre-edit actual (dimmed) + model predictions
        def _build_panel(pred_obs: np.ndarray) -> np.ndarray:
            panel = np.zeros((total_frames, R), dtype=np.float32)
            panel[: c4.edit_frame] = np.clip(c4.viz_obs_pre_edit[i], 0, 1) * 0.35
            panel[c4.edit_frame :] = np.clip(pred_obs, 0, 1)
            return panel

        steered_full = _build_panel(c4.viz_steered[i])
        unsteered_full = _build_panel(c4.viz_unsteered[i])

        fig_3 = plt.figure(figsize=(18, 5.5), facecolor=_DARK_BG_HEX)
        fig_3.suptitle(
            f"Sample {i}  —  counterfactual controllability  "
            f"(edit at frame {c4.edit_frame})",
            color=_DARK_TEXT_COLOR, fontsize=11, y=0.99,
        )
        fig_3.subplots_adjust(
            left=0.05, right=0.97, top=0.90, bottom=0.12, wspace=0.18,
        )

        ax_gt = fig_3.add_subplot(1, 3, 1)
        ax_st = fig_3.add_subplot(1, 3, 2)
        ax_un = fig_3.add_subplot(1, 3, 3)

        for ax, obs_img, ttl in zip(
            [ax_gt, ax_st, ax_un],
            [gt_obs_full, steered_full, unsteered_full],
            ["GT (full sequence)", "steered rollout", "unsteered rollout"],
        ):
            _style_ax_dark(ax)
            ax.imshow(
                obs_img, aspect="auto", origin="upper", interpolation="nearest",
                cmap="gray", vmin=0, vmax=1,
            )
            ax.axhline(
                c4.edit_frame - 0.5, color="#fa8850", linewidth=1.2,
                linestyle="--", alpha=0.7,
            )
            ax.set_title(ttl, color=_DARK_TEXT_COLOR, fontsize=10)
            ax.set_xlabel("ray position", color=_DARK_TEXT_COLOR, fontsize=9)
            ax.set_ylabel("frame", color=_DARK_TEXT_COLOR, fontsize=9)

        plt.tight_layout()
        figs[f"controllability_{i}"] = fig_3

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
            "next_step_mse_std": c1.metrics.std_mse,
        }
    if "c2" in results:
        c2 = results["c2"]
        metrics["criterion2"] = {
            "linear_overall_mse": c2.recovery_linear.overall_mse,
            "mlp_overall_mse": c2.recovery_mlp.overall_mse,
        }
    if "c3" in results:
        c3 = results["c3"]
        metrics["criterion3"] = {
            "linear_coherence_mean": c3.coherence_metrics.mean_score,
            "mlp_coherence_mean": c3.mlp_coherence_metrics.mean_score,
            "gt_coherence_mean": float(c3.gt_per_sample_scores.mean()),
            "linear_coherence_std": c3.coherence_metrics.std_score,
            "mean_jump_ratio": c3.coherence_metrics.mean_jump_ratio,
        }
    if "c4" in results:
        c4 = results["c4"]
        metrics["criterion4"] = {
            "steered_mse": c4.ctrl_metrics.steered_mse,
            "unsteered_mse": c4.ctrl_metrics.unsteered_mse,
            "injection_error": c4.ctrl_metrics.injection_error,
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  metrics → {output_dir / 'metrics.json'}")


def save_config(cfg: EvalConfig, s: SetupResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "eval_config": cfg.__dict__,
        "checkpoint": {
            "epoch": s.ckpt_info["epoch"],
            "val_loss": s.ckpt_info["val_loss"],
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
            raise ValueError(
                "Criterion 2 requires criterion 1 (needs internal_states_tf)"
            )
        results["c2"] = run_criterion2(cfg, s, results["c1"])
    if 3 in cfg.criteria:
        if "c2" not in results:
            raise ValueError(
                "Criterion 3 requires criterion 2 (needs linear_extractor)"
            )
        results["c3"] = run_criterion3(cfg, s, results["c2"])
    if 4 in cfg.criteria:
        if "c2" not in results:
            raise ValueError(
                "Criterion 4 requires criterion 2 (needs linear_extractor)"
            )
        results["c4"] = run_criterion4(cfg, s, results["c2"])

    return results
