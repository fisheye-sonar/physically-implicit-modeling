"""Experiment orchestration for the Othello-GPT method port.

Thin glue only: loads the run, fits the per-residual-point probes, assembles the
edit arms, and scores them with the canonical §4 metrics. The *method* lives in
`othello_probe.py`; the *metrics* live in `scripts/editability_metrics.py`. Nothing
is re-derived here (see `harness/ANALYSIS.md` §1).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "scripts"))
sys.path.insert(0, str(_HERE.parent))

import othello_probe as op  # noqa: E402
from editability_metrics import (  # noqa: E402
    build_edit_zones,
    edit_scorecard,
    fidelity_ratio,
)
from pim.world_models import load_checkpoint, load_dataset  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_OBJ = 2
K = 15  # post-edit rollout steps, repo standard
N_CTX = 6  # noisy context frames shown above the edit line
LATE_T = 15  # "late" frames = t >= 15, the filter-converged regime

# Probe targets. `pos` is the 4-dim position read-out the edit actually drives;
# `full` adds velocity so the probe carries the whole world state at once.
TARGETS = {
    "pos": ["obj0 x", "obj0 y", "obj1 x", "obj1 y"],
    "full": [
        "obj0 x",
        "obj0 y",
        "obj1 x",
        "obj1 y",
        "obj0 vx",
        "obj0 vy",
        "obj1 vx",
        "obj1 vy",
    ],
}


def residual_point_label(ell: int, n_layers: int = 4) -> str:
    """Self-describing axis label — never a bare integer (`harness/STYLE.md` §5)."""
    if ell == 0:
        return "0 · encoder port"
    if ell == n_layers:
        return f"{ell} · last (decoder input)"
    return f"{ell} · block {ell} input"


@dataclass
class Bundle:
    model: object
    test: object
    edits: object
    sim: dict
    info: object


def load(run: str = "W16", dataset: str = "4_fixed_refl_inview") -> Bundle:
    model, info = load_checkpoint(
        f"runs/transformers/{run}/best_model.pt", device=DEVICE
    )
    model.eval()
    b = load_dataset(f"datasets/{dataset}", n_obj_keep=N_OBJ)
    return Bundle(model, b.test, b.edits, b.test.config["dataset"]["sim"], info)


# ── probes ────────────────────────────────────────────────────────────────────


def probe_table(
    bundle: Bundle,
    *,
    n_seq: int = 1500,
    hidden: int = 512,
    epochs: int = 200,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Fit linear and MLP probes at every residual point, for both targets.

    Held out by **sequence** (80/20), not by frame — see `othello_probe` docstring
    and `research/GOTCHAS.md` (2026-08-14).
    """
    model, test = bundle.model, bundle.test
    obs = test.obs[:n_seq]
    T = obs.shape[1]
    R = op.collect_residuals(model, obs, batch=128)
    pos = test.positions[:n_seq, :T, :N_OBJ, :].astype(np.float32).reshape(n_seq, T, 4)
    with h5py.File(test.h5_path, "r") as f:
        vel = f["velocities"][:n_seq, :T, :N_OBJ, :].astype(np.float32)
    vel = vel.reshape(n_seq, T, 4)
    Y = {"pos": pos, "full": np.concatenate([pos, vel], -1)}
    vis = test.is_visible[:n_seq, :T, :N_OBJ].all(axis=2)
    late = np.zeros_like(vis)
    late[:, LATE_T:] = True

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_seq)
    ntr = int(0.8 * n_seq)
    tr, te = perm[:ntr], perm[ntr:]

    probes: dict = {}
    stats: list[dict] = []
    n_points = R.shape[0]
    for tname, Yt in Y.items():
        for ell in range(n_points):
            X = R[ell]
            for hid, fam in ((None, "linear"), (hidden, f"MLP ({hidden} hidden)")):
                p, s = op.fit_probe(
                    X[tr][vis[tr]],
                    Yt[tr][vis[tr]],
                    X[te][vis[te]],
                    Yt[te][vis[te]],
                    hidden=hid,
                    epochs=epochs,
                    device=DEVICE,
                    seed=seed,
                )
                # late-t is the repo's reporting convention for velocity
                _, s_late = op.fit_probe(
                    X[tr][(vis & late)[tr]],
                    Yt[tr][(vis & late)[tr]],
                    X[te][(vis & late)[te]],
                    Yt[te][(vis & late)[te]],
                    hidden=hid,
                    epochs=epochs,
                    device=DEVICE,
                    seed=seed,
                )
                stats.append(
                    dict(
                        target=tname,
                        point=ell,
                        family=fam,
                        r2=s["r2"],
                        r2_late=s_late["r2"],
                        r2_insample=s["r2_insample"],
                        rmse=s["rmse"],
                        per_dim_r2=s["per_dim_r2"],
                        per_dim_r2_late=s_late["per_dim_r2"],
                    )
                )
                if hid is not None:
                    probes[(tname, ell)] = p
    return probes, stats


# ── edit arms ─────────────────────────────────────────────────────────────────


@dataclass
class EditSetup:
    state: object
    state_oracle: object
    zones: object
    gt_roll: np.ndarray
    ctx: np.ndarray
    tgt_cx: np.ndarray
    ghost_cx: np.ndarray
    change_mask: dict
    target_values: dict
    x0: dict
    n: int


def _centroid(m: np.ndarray) -> np.ndarray:
    out = np.full(len(m), np.nan)
    for i in range(len(m)):
        idx = np.where(m[i])[0]
        if idx.size:
            out[i] = idx.mean()
    return out


def edit_setup(bundle: Bundle, n_edit: int = 256) -> EditSetup:
    model, edits, sim = bundle.model, bundle.edits, bundle.sim
    ef = edits.edit_frame
    n = min(n_edit, edits.n_samples)
    oe = edits.edit_object[:n].astype(int)

    state = model.state_from_obs(
        torch.from_numpy(edits.obs[:n, :ef]).float().to(DEVICE)
    )
    # Oracle observation: one extra teacher-forced frame, the REAL (noisy) post-edit
    # observation. Its rollout therefore LEADS THE OTHERS BY ONE FRAME — label it,
    # never re-align the other arms to it.
    state_oracle = model.state_from_obs(
        torch.from_numpy(edits.obs[:n, : ef + 1]).float().to(DEVICE)
    )

    gt_roll = edits.clean_obs[:n, ef : ef + K].astype(np.float32)
    with h5py.File(edits.h5_path, "r") as f:
        pre_vel = f["velocities"][:n, ef - 1, :N_OBJ, :].astype(np.float32)
    zones = build_edit_zones(
        pre_pos=edits.positions[:n, ef - 1, :N_OBJ, :].astype(np.float32),
        tgt_pos=edits.positions[:n, ef, :N_OBJ, :].astype(np.float32),
        pre_vel=pre_vel,
        edit_object=oe,
        sim=sim,
        n_obj=N_OBJ,
        traj_pos=edits.positions[:n, ef : ef + K, :N_OBJ, :].astype(np.float32),
        gt_edited_traj=gt_roll,
    )

    tgt_pos = edits.positions[:n, ef, :N_OBJ, :].astype(np.float32).reshape(n, 4)
    change_mask, target_values = {}, {}
    for tname, names in TARGETS.items():
        d_out = len(names)
        cm = np.zeros((n, d_out), bool)
        tv = np.zeros((n, d_out), np.float32)
        for i, o in enumerate(oe):
            cm[i, 2 * o : 2 * o + 2] = True  # only the edited object's POSITION moves
        tv[:, :4] = tgt_pos
        change_mask[tname] = cm
        target_values[tname] = torch.tensor(tv, device=DEVICE)

    rs = model.residual_stack(state)
    x0 = {ell: rs[ell][:, -1] for ell in range(rs.shape[0])}
    return EditSetup(
        state=state,
        state_oracle=state_oracle,
        zones=zones,
        gt_roll=gt_roll,
        ctx=edits.obs[:n, ef - N_CTX : ef].astype(np.float32),
        tgt_cx=_centroid(zones.target),
        ghost_cx=_centroid(zones.ghost),
        change_mask=change_mask,
        target_values=target_values,
        x0=x0,
        n=n,
    )


def free_rollout(model, state, steps: int = K) -> np.ndarray:
    """Plain free-run. `predict_step` decodes *and* advances, so it is called once
    per step — decoding separately first would emit step 0 twice."""
    out = []
    s = state
    with torch.no_grad():
        for _ in range(steps):
            p, s = model.predict_step(s)
            out.append(p)
    return torch.stack(out, 1).cpu().numpy()


def run_arms(
    bundle: Bundle,
    setup: EditSetup,
    probes: dict,
    target: str,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    beta: float = 1.0,
    optimizer: str = "adam",
    start_layers=(0, 1, 2, 3, 4),
) -> tuple[dict, dict, dict]:
    """Every arm's rollout + scorecard for one probe target.

    Returns (rollouts, cards, records). `cards` carries `edit_index_by_step` — never
    strip it when serialising (`harness/ANALYSIS.md` §1).
    """
    model = bundle.model
    P = {ell: probes[(target, ell)] for ell in range(len(setup.x0))}
    specs = {
        ell: op.build_edit_spec(
            P[ell],
            setup.x0[ell],
            setup.change_mask[target],
            setup.target_values[target],
            beta=beta,
        )
        for ell in P
    }

    rolls = {"Unsteered": free_rollout(model, setup.state)}
    records: dict = {}
    for ls in start_layers:
        rec: dict = {}
        rolls[f"from {residual_point_label(ls)}"] = (
            op.rollout_with_sequential_intervention(
                model,
                setup.state,
                P,
                specs,
                ls,
                K,
                alpha=alpha,
                n_steps=n_steps,
                optimizer=optimizer,
                record=rec,
            )
            .cpu()
            .numpy()
        )
        records[ls] = rec
    rolls["Oracle observation"] = free_rollout(model, setup.state_oracle)

    cards = {k: edit_scorecard(v, setup.zones, setup.gt_roll) for k, v in rolls.items()}
    for k in cards:
        cards[k]["fidelity_ratio"] = fidelity_ratio(cards[k], cards["Unsteered"])
    return rolls, cards, records


def representative_samples(
    teleport: np.ndarray, k: int = 4, seed: int = 0
) -> list[int]:
    """Sample indices spread across the teleport-size range, one per quantile band.

    Picking the k LARGEST teleports flatters an editor: measured 2026-08-18, the four
    largest-teleport episodes sat at the **98th percentile** of the Edit Index
    distribution (+0.07 against a −0.54 mean), because this editor's effect grows with
    teleport size while the unsteered baseline is flat. A qualitative panel selected that
    way shows the best case and reads as the typical one.
    """
    order = np.argsort(teleport)
    bands = np.array_split(order, k)
    rng = np.random.default_rng(seed)
    return [int(b[len(b) // 2]) if len(b) else int(rng.choice(order)) for b in bands]


def random_samples(n: int, k: int = 4, seed: int = 0) -> list[int]:
    """`k` episode indices drawn uniformly at random — the DEFAULT for any qualitative panel.

    Seeded so the panel is reproducible. Use this unless there is a stated reason not to; if a
    panel deliberately shows extreme cases, say so in the figure title (`harness/STYLE.md` §2).
    """
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n, size=min(k, n), replace=False))
