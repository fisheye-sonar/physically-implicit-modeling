"""Autonomous RSSM refinement sweep.

Goal (engineering, not science): give the RSSM a good-faith effort at becoming a
strong *predictor* of observations under the project's premise — trained only on
the predictive/generative task (ELBO over observations), never on GT positions.
We therefore optimise long-/near-horizon observation accuracy, and read
recoverability + rollout coherence as *diagnostics* (do they fall out of better
prediction?), not as objectives.

Design (orchestrator-friendly, survives the 10-min foreground ceiling):
  * This script is launched DETACHED (`setsid nohup ... &`) and owns the whole
    sweep. Each arm trains via a `train_rssm.py` subprocess (fresh CUDA context),
    then is evaluated here. Progress is machine-readable: `results.json` is
    rewritten after every arm; `sweep.log` carries human-readable lines.
  * Idempotent / resumable: arms already marked `done` in results.json are
    skipped; a partially-done arm is wiped and retrained. So a crash or a
    session restart loses at most the in-flight arm.
  * Time-guarded: before each arm, if elapsed > --max-hours, stop the ladder and
    still run the FINAL long arm on the best config so far.

Predictive eval is DETERMINISTIC for the RSSM: we set `model.sample = False` so
imagined rollouts use the prior mean (standard Dreamer eval) rather than being
penalised by per-step sampling noise.

Usage
-----
    python scripts/sweep_rssm.py \
        --dataset-dir datasets/4_fixed_refl_inview \
        --out runs/rssm_sweep \
        --max-hours 15
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# Allow running from repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pim.eval import (  # noqa: E402
    autoregressive_rollouts,
    collect_rollouts,
    decode_states_multi,
    eval_horizon_mse,
    eval_single_step,
    eval_trajectory_coherence,
    fit_probes,
    eval_recovery_multi,
    teacher_force,
)
from pim.extractors import LinearExtractor, MLPExtractor, ProbeSpec, StateDefinition  # noqa: E402
from pim.extractors.matching import identity_mse  # noqa: E402
from pim.world_models import load_checkpoint, load_dataset, make_test_loader  # noqa: E402

# ── Config ladder ───────────────────────────────────────────────────────────
# Each arm is a full, explicit model+train config (no "best-so-far" chaining, so
# arms are independent and the sweep is robust to interruption). One backbone B
# (capacity + lr jump together — the expected big win); the other arms vary ONE
# knob against B so each result is interpretable. The FINAL arm is the only
# adaptive one: it retrains the best arm's config for many more epochs.

SCREEN_EPOCHS = 220
FINAL_EPOCHS = 500

_BACKBONE = dict(
    embed_dim=128, det_size=256, stoch_size=48, hidden_dim=256,
    enc_layers=1, dec_layers=1,
    lr=1e-3, weight_decay=1e-4,
    kl_weight=1.0, kl_warmup_epochs=10, free_nats=3.0, kl_balance_alpha=0.0,
    n_epochs=SCREEN_EPOCHS,
)


def _arm(name: str, **overrides) -> dict:
    cfg = dict(_BACKBONE)
    cfg.update(overrides)
    cfg["name"] = name
    return cfg


LADDER: list[dict] = [
    # Anchor: the current good RSSM (run3) config, just trained longer. Reference
    # point — everything else should beat this.
    _arm("anchor", embed_dim=128, det_size=94, stoch_size=34, hidden_dim=128, lr=3e-4),
    # The backbone: bigger + lr 1e-3 together.
    _arm("backbone"),
    # Isolate the lr bump.
    _arm("bb_lr3e-4", lr=3e-4),
    # More stochastic capacity.
    _arm("bb_stoch64", stoch_size=64),
    # DreamerV2 KL balancing (train the prior harder).
    _arm("bb_klb0.8", kl_balance_alpha=0.8),
    # KL-pressure sweep (free_nats): richer vs looser latent.
    _arm("bb_fn1", free_nats=1.0),
    _arm("bb_fn6", free_nats=6.0),
    # Deeper encoder + decoder (observation fidelity).
    _arm("bb_deep", enc_layers=2, dec_layers=2),
    # Larger still.
    _arm("bb_big", embed_dim=256, det_size=384, stoch_size=64, hidden_dim=384),
]

_MODEL_KEYS = ("embed_dim", "det_size", "stoch_size", "hidden_dim", "enc_layers", "dec_layers")
_TRAIN_KEYS = ("lr", "weight_decay", "kl_weight", "kl_warmup_epochs", "free_nats",
               "kl_balance_alpha", "n_epochs")


# ── Evaluation ──────────────────────────────────────────────────────────────


def evaluate_predictor(
    ckpt_path: str,
    test,
    device: str,
    *,
    n_context: int = 10,
    n_roll_samples: int = 2000,
    n_probe_samples: int = 5000,
    n_coherence: int = 500,
    n_obj: int = 2,
) -> dict:
    """Deterministic predictive eval + recoverability/coherence diagnostics.

    Primary metric = near-horizon clean-obs MSE (mean over H=1..5). Clean obs is
    the noiseless signal, so it isolates the learnable part from the irreducible
    observation-noise floor. Horizon arrays are returned in full so the
    near/mid/long bands and the persistence comparison are all reconstructable.
    """
    model, info = load_checkpoint(ckpt_path, device=device)
    if hasattr(model, "sample"):
        model.sample = False  # deterministic prior-mean rollout for fair prediction

    T = test.T_frames
    n_rollout = T - n_context

    # ── Next-step (teacher forced, deterministic) ──
    loader = make_test_loader(test, batch_size=512, num_workers=0)
    preds_tf, states_tf = teacher_force(model, loader, device=device)
    next_clean = float(eval_single_step(test.clean_obs, preds_tf).mean_mse)
    next_noisy = float(eval_single_step(test.obs, preds_tf).mean_mse)

    # ── Open-loop imagined rollout (the dynamics test) ──
    obs_roll = autoregressive_rollouts(
        model, test.obs[:n_roll_samples], n_context=n_context, device=device,
    )
    horizon_clean = eval_horizon_mse(test.clean_obs[:n_roll_samples], obs_roll, n_context)
    horizon_noisy = eval_horizon_mse(test.obs[:n_roll_samples], obs_roll, n_context)

    # Persistence baseline: freeze the last context frame for the whole rollout.
    last_ctx = test.obs[:n_roll_samples, n_context - 1:n_context, :]  # (N,1,R)
    persist = np.repeat(last_ctx, n_rollout, axis=1)
    persist_clean = eval_horizon_mse(test.clean_obs[:n_roll_samples], persist, n_context)

    def band(arr, lo, hi):
        return float(np.mean(arr[lo:hi]))

    # ── Recoverability diagnostic (probes; NOT an objective) ──
    recovery = {}
    try:
        s = states_tf[:n_probe_samples]
        env = test.positions[:n_probe_samples, :-1, :n_obj, :]
        mask = test.is_visible[:n_probe_samples, :-1, :n_obj].all(axis=2)
        state_def = StateDefinition(
            name="positions", state_shape=(n_obj, 2),
            extract_fn=lambda b: b["positions"],
        )
        probes = [
            ProbeSpec(name="linear",
                      probe=LinearExtractor(model.hidden_size, state_def, use_lstsq=True)),
            ProbeSpec(name="MLP",
                      probe=MLPExtractor(model.hidden_size, state_def,
                                         mlp_hidden=256, n_epochs=30, lr=5e-3)),
        ]
        fit_probes(probes, s, env, mask=mask, loss_fn=identity_mse, device=device)
        rec = eval_recovery_multi(probes, s, env, mask=mask, device=device)
        recovery = {k: float(v.overall_mse) for k, v in rec.items()}
    except Exception as e:  # diagnostic only — never fail the arm on it
        recovery = {"error": repr(e)}

    # ── Rollout coherence diagnostic (smoothness of decoded positions) ──
    coherence = {}
    try:
        _, h_roll, _ = collect_rollouts(
            model, test.obs[:n_coherence], n_context=n_context,
            n_rollout=n_rollout, device=device,
        )
        dec = decode_states_multi(probes, h_roll, device=device)
        cm = eval_trajectory_coherence(dec["MLP"])
        coherence = {"smoothness": float(cm.mean_score), "jump_ratio": float(cm.mean_jump_ratio)}
    except Exception as e:
        coherence = {"error": repr(e)}

    return {
        "epoch": int(info.epoch),
        "val_loss": float(info.val_loss),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "next_step_clean_mse": next_clean,
        "next_step_noisy_mse": next_noisy,
        "near_h_clean_mse": band(horizon_clean, 0, 5),     # PRIMARY (H=1..5)
        "mid_h_clean_mse": band(horizon_clean, 5, 15),     # H=6..15
        "long_h_clean_mse": band(horizon_clean, n_rollout - 5, n_rollout),  # last 5
        "horizon_clean": [float(x) for x in horizon_clean],
        "horizon_noisy": [float(x) for x in horizon_noisy],
        "persistence_clean": [float(x) for x in persist_clean],
        "persist_near_h": band(persist_clean, 0, 5),
        "recovery": recovery,
        "coherence": coherence,
        "n_context": n_context,
    }


# ── Sweep driver ────────────────────────────────────────────────────────────


def _train_arm(arm: dict, dataset_dir: str, arm_dir: Path) -> None:
    """Train one arm via a fresh train_rssm.py subprocess."""
    shutil.rmtree(arm_dir, ignore_errors=True)
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_rssm.py"),
        "--dataset-dir", dataset_dir,
        "--run-dir", str(arm_dir),
        "--run-name", arm["name"],
        "--n-epochs", str(arm["n_epochs"]),
        "--lr", str(arm["lr"]),
        "--weight-decay", str(arm["weight_decay"]),
        "--kl-weight", str(arm["kl_weight"]),
        "--kl-warmup-epochs", str(arm["kl_warmup_epochs"]),
        "--free-nats", str(arm["free_nats"]),
        "--kl-balance-alpha", str(arm["kl_balance_alpha"]),
        "--embed-dim", str(arm["embed_dim"]),
        "--det-size", str(arm["det_size"]),
        "--stoch-size", str(arm["stoch_size"]),
        "--hidden-dim", str(arm["hidden_dim"]),
        "--enc-layers", str(arm["enc_layers"]),
        "--dec-layers", str(arm["dec_layers"]),
    ]
    subprocess.run(cmd, check=True)


def _load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_results(path: Path, results: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(path)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ckpt_epoch(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu")["epoch"])
    except Exception:
        return -1


def _run_one(arm: dict, dataset_dir: str, out: Path, results: dict,
             results_path: Path, device: str) -> None:
    name = arm["name"]
    if results.get(name, {}).get("status") == "done":
        _log(f"skip {name} (already done)")
        return
    arm_dir = out / name
    results[name] = {"status": "running", "config": arm}
    _save_results(results_path, results)
    t0 = time.time()
    try:
        # Salvage already-trained arms across restarts: if latest.pt is already at
        # the target epoch, skip training and just (re-)evaluate it.
        latest = arm_dir / "latest.pt"
        if latest.exists() and _ckpt_epoch(latest) >= arm["n_epochs"]:
            _log(f"REUSE {name}: latest.pt already at epoch {_ckpt_epoch(latest)} — eval only")
        else:
            _log(f"TRAIN {name}: {json.dumps({k: arm[k] for k in _MODEL_KEYS + _TRAIN_KEYS})}")
            _train_arm(arm, dataset_dir, arm_dir)
        # Evaluate the fully-trained checkpoint (latest.pt), which is invariant to
        # the KL warm-up / free-nats schedule that can mislead best-by-total-loss.
        if not latest.exists():
            raise FileNotFoundError(f"no checkpoint at {latest}")
        _log(f"EVAL {name}")
        metrics = evaluate_predictor(str(latest), TEST, device)
        metrics["train_minutes"] = round((time.time() - t0) / 60, 1)
        results[name] = {"status": "done", "config": arm, "metrics": metrics}
        _log(f"DONE {name}  near_h_clean={metrics['near_h_clean_mse']:.5f}  "
             f"next={metrics['next_step_clean_mse']:.5f}  "
             f"rec_mlp={metrics['recovery'].get('MLP', float('nan')):.4f}  "
             f"({metrics['train_minutes']}min)")
    except Exception:
        results[name] = {"status": "failed", "config": arm, "error": traceback.format_exc()}
        _log(f"FAILED {name}\n{traceback.format_exc()}")
    _save_results(results_path, results)


def _best_arm(results: dict) -> tuple[str, dict] | tuple[None, None]:
    done = {k: v for k, v in results.items()
            if v.get("status") == "done" and "metrics" in v and not k.startswith("FINAL")}
    if not done:
        return None, None
    best = min(done.items(), key=lambda kv: kv[1]["metrics"]["near_h_clean_mse"])
    return best[0], best[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-hours", type=float, default=15.0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny 1-arm 2-epoch run to validate the full pipeline")
    p.add_argument("--ladder-json", default="",
                   help="JSON file with a list of (possibly partial) arm dicts; each is "
                        "merged over the backbone defaults. Overrides the built-in ladder.")
    a = p.parse_args()

    global LADDER, FINAL_EPOCHS
    if a.smoke:
        LADDER = [_arm("smoke", det_size=64, stoch_size=16, hidden_dim=64, n_epochs=2)]
        FINAL_EPOCHS = 2
    elif a.ladder_json:
        spec = json.loads(Path(a.ladder_json).read_text())
        LADDER = [_arm(d.pop("name"), **d) for d in spec]
        _log(f"loaded ladder from {a.ladder_json}: {[x['name'] for x in LADDER]}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = _load_results(results_path)

    # Load test split once; stash globally so _run_one can reach it.
    global TEST
    TEST = load_dataset(a.dataset_dir, n_obj_keep=2, require_edits=False).test
    _log(f"loaded test split: {TEST.n_samples} samples T={TEST.T_frames} R={TEST.obs_res}")

    start = time.time()
    for arm in LADDER:
        if (time.time() - start) / 3600 > a.max_hours:
            _log(f"time guard hit ({a.max_hours}h) — stopping ladder, going to FINAL")
            break
        _run_one(arm, a.dataset_dir, out, results, results_path, a.device)

    # FINAL adaptive arm: retrain the best config for many more epochs.
    bname, bval = _best_arm(results)
    if bname is not None:
        final = dict(bval["config"])
        final["name"] = "FINAL"
        final["n_epochs"] = FINAL_EPOCHS
        _log(f"FINAL based on best arm '{bname}' "
             f"(near_h_clean={bval['metrics']['near_h_clean_mse']:.5f})")
        _run_one(final, a.dataset_dir, out, results, results_path, a.device)

    # Summary table sorted by primary metric.
    _log("=" * 78)
    _log("SWEEP SUMMARY (sorted by near-horizon clean MSE, H=1..5)")
    rows = [(k, v["metrics"]) for k, v in results.items()
            if v.get("status") == "done" and "metrics" in v]
    rows.sort(key=lambda kv: kv[1]["near_h_clean_mse"])
    for k, m in rows:
        _log(f"  {k:14s} near={m['near_h_clean_mse']:.5f} mid={m['mid_h_clean_mse']:.5f} "
             f"long={m['long_h_clean_mse']:.5f} next={m['next_step_clean_mse']:.5f} "
             f"rec_mlp={m['recovery'].get('MLP', float('nan')):.4f} "
             f"persist_near={m['persist_near_h']:.5f} ep={m['epoch']}")
    _log("DONE — sweep complete.")


if __name__ == "__main__":
    TEST = None  # set in main()
    main()
