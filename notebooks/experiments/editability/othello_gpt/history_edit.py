"""Backward history rewrite: apply the edit to the WHOLE observed history, not the latent.

Every editor in this thread writes to one state at one frame and the dynamics reject it. This
module tests the opposite move. To teleport object `k` by a displacement `delta` at the edit
frame, it applies **the same `delta` to every prior frame**, rebuilding each observation from the
model's **own decoded positions** — and then teacher-forces the model on that rewritten history.

Why this should work in principle
---------------------------------
Objects move at constant velocity in this world, so translating one object's whole trajectory by
a constant `delta` is itself a valid trajectory: same velocity, offset position. The rewritten
history is therefore a physically consistent world in which the object simply *was* somewhere
else all along — not a single inconsistent frame the dynamics have to absorb. If the model's
belief is driven by the observations (and the `input_grad_steering` and freeze-time results say
it is), this is the intervention its dynamics should honour.

No ground truth is used
-----------------------
Positions come from the **probe's read-out of the model's own residual stream**, never from the
dataset. Rendering needs only the object radius and reflectivities, which on a
`fixed_reflectivities` dataset are **world constants identical for every episode**, not
per-episode state (`editability_metrics.object_constants`). The displacement itself is computed
from decoded quantities: where the probe thinks the object is at `ef-1`, plus the per-frame step
estimated by finite differences of the decoded track.

What this trades away, stated up front
--------------------------------------
It uses the **renderer**. That is a real departure from a pure latent intervention and it is why
this is a separate notebook rather than another arm in the editor gallery: it answers "will the
dynamics honour a consistent history?", not "can we find the right direction in latent space".
The `delta = 0` reconstruction control is what makes the comparison honest — it measures the cost
of round-tripping through decode-and-re-render before any edit is applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[4] / "scripts"))
sys.path.insert(0, str(_HERE.parent))

import othello_probe as op  # noqa: E402
from editability_metrics import object_constants, sim_config_from  # noqa: E402
from pim.simulator.renderer import render_frame  # noqa: E402

N_OBJ = 2


@torch.no_grad()
def decode_history_positions(
    model, probe, obs: np.ndarray, point: int, batch: int = 128
):
    """(n, T, n_obj, 2) positions read by `probe` from the residual stream at every frame.

    This is the model's *own* belief about where the objects are — the only source of position
    information this experiment is allowed to use.
    """
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(dev)
        tokens = model.embed(o)
        _, resids = model._run(
            tokens, model._seq_mask(o.shape[1], dev), want_resid=True
        )
        x = resids[point]  # (B, T, d)
        B, T, D = x.shape
        pred = probe(x.reshape(B * T, D)).reshape(B, T, -1)
        out.append(pred[..., : N_OBJ * 2].cpu().numpy())
    return np.concatenate(out, 0).reshape(len(obs), obs.shape[1], N_OBJ, 2)


def estimate_step(track: np.ndarray, n_fit: int = 10) -> np.ndarray:
    """(n, n_obj, 2) per-frame displacement, from finite differences of the decoded track.

    Velocity is constant in this world, so the mean first difference over the last `n_fit`
    frames is the natural estimator — and it is far more reliable than reading velocity off the
    probe, whose velocity dimensions barely rise above chance here.
    """
    d = np.diff(track[:, -n_fit:], axis=1)
    return d.mean(axis=1)


def edit_delta(
    track: np.ndarray, target_pos: np.ndarray, edit_object: np.ndarray, n_fit: int = 10
) -> np.ndarray:
    """(n, 2) displacement to apply to EVERY frame of the edited object's decoded track.

    Chosen so the object lands on `target_pos` at the edit frame: the decoded track says it
    would otherwise be at `track[:, -1] + step`, so `delta = target − (track[:, -1] + step)`.
    """
    step = estimate_step(track, n_fit)
    idx = np.arange(len(track))
    k = edit_object.astype(int)
    predicted_next = track[idx, -1, k] + step[idx, k]
    return target_pos[idx, k] - predicted_next


def render_history(
    pos: np.ndarray, sim: dict, *, noise_std: float = 0.0, seed: int = 0
) -> np.ndarray:
    """(n, T, R) clean renders of a position history, optionally with matched observation noise.

    `noise_std > 0` reproduces the sensing noise the model was trained and teacher-forced on;
    the model has never seen a noiseless observation, so both variants are worth running.
    """
    n, T = pos.shape[0], pos.shape[1]
    cfg = sim_config_from(sim, N_OBJ)
    rad, refl = object_constants(sim, N_OBJ)
    R = int(sim["obs_res"])
    out = np.zeros((n, T, R), np.float32)
    for i in range(n):
        for t in range(T):
            _, _, inten = render_frame(pos[i, t].astype(np.float32), rad, refl, cfg)
            out[i, t] = inten
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        out = np.clip(
            out + rng.normal(0, noise_std, out.shape).astype(np.float32), 0.0, 1.0
        )
    return out


def shifted_history(
    track: np.ndarray, delta: np.ndarray, edit_object: np.ndarray
) -> np.ndarray:
    """(n, T, n_obj, 2) — the decoded track with the edited object translated by `delta`
    at EVERY frame. The other object is left exactly where the probe read it."""
    out = track.copy()
    idx = np.arange(len(track))
    k = edit_object.astype(int)
    out[idx, :, k] = out[idx, :, k] + delta[:, None, :]
    return out


@torch.no_grad()
def rollout_from_history(
    model, frames: np.ndarray, steps: int, device: str
) -> np.ndarray:
    """Teacher-force the model on `frames`, then free-run `steps` — the same convention as
    every other arm, so step 0 decodes the edit frame."""
    st = model.state_from_obs(torch.from_numpy(frames).float().to(device))
    out = []
    s = st
    for _ in range(steps):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


def visibility_report(pos: np.ndarray, sim: dict) -> dict:
    """How often a trajectory leaves the visible frustum.

    Translating a track by a constant `delta` can push the object out of view, which would make
    the rewritten history unrenderable-as-intended rather than merely different. Reported so the
    result is not silently confounded by it.

    Uses the simulator's own `frustum_half_width` rather than re-deriving the test — an earlier
    hand-rolled version halved `x_near`, which is *already* a half-width, and reported 28% of
    GROUND-TRUTH frames as out of view on an `always_in_frustum` dataset.
    """
    from pim.simulator import frustum_half_width

    cfg = sim_config_from(sim, N_OBJ)
    r = float(sim["radius"])
    y = np.clip(pos[..., 1], float(sim["y_near"]), float(sim["y_far"]))
    x_lim = np.vectorize(lambda yy: float(frustum_half_width(yy, cfg)))(y) - r
    inside = (
        (pos[..., 1] >= float(sim["y_near"]))
        & (pos[..., 1] <= float(sim["y_far"]))
        & (np.abs(pos[..., 0]) <= x_lim)
    )
    return {
        "frac_frames_outside": float(1.0 - inside.mean()),
        "frac_episodes_any_outside": float((~inside).any(axis=(1, 2)).mean()),
    }


# ── Renderer-free history editing: the Othello write applied at EVERY frame ────
#
# Nothing below renders anything, reads ground truth, or consults an oracle. The only
# inputs are the model, the MLP probe fit on the test split, and the displacement `δ`
# derived from the model's own decoded track.


def _history_targets(
    track: np.ndarray, delta: np.ndarray, edit_object: np.ndarray, beta: float
):
    """(values, weight) per history frame, both `(n, n_hist, 4)`.

    The edited object is sent to its decoded position **plus `δ` at every frame**; the
    other object is held at the position the probe already reads there. This is the
    paper's `B' = B except at the edited site`, applied at every frame instead of one.
    """
    tgt = track.copy()
    idx = np.arange(len(track))
    k = edit_object.astype(int)
    tgt[idx, :, k] = tgt[idx, :, k] + delta[:, None, :]
    values = tgt.reshape(len(track), track.shape[1], N_OBJ * 2)
    weight = np.full_like(values, beta)
    for i, kk in enumerate(k):
        weight[i, :, 2 * kk : 2 * kk + 2] = 1.0
    return values, weight


@torch.no_grad()
def activation_history_edit_rollout(
    model,
    state,
    probes: dict,
    track: np.ndarray,
    delta: np.ndarray,
    edit_object: np.ndarray,
    start_layer: int,
    steps: int,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    beta: float = 1.0,
    optimizer: str = "adam",
    record: dict | None = None,
) -> np.ndarray:
    """The Othello write applied at **every history position**, at every residual point
    from `start_layer` on — the paper's Figure 2C schedule widened from one timestep to
    the whole window.

    The transformer's residual stream at window position `t` is its representation of
    frame `t`, so rewriting the history in latent space means writing every position and
    re-applying at each subsequent layer, because every block recomputes the stream.
    """
    dev = next(model.parameters()).device
    S = model.state_span
    n, n_hist = track.shape[0], track.shape[1]
    off = (
        S - n_hist
    )  # buffer is right-aligned, so frame t sits at buffer position off+t

    vals_np, wt_np = _history_targets(track, delta, edit_object, beta)
    vals = torch.tensor(
        vals_np.reshape(n * n_hist, -1), device=dev, dtype=torch.float32
    )
    wts = torch.tensor(wt_np.reshape(n * n_hist, -1), device=dev, dtype=torch.float32)

    def hook(layer: int, x: torch.Tensor) -> torch.Tensor:
        if layer < start_layer or layer not in probes:
            return x
        probe = probes[layer]
        cur = x[:, off:].reshape(n * n_hist, x.shape[-1])
        spec = op.EditSpec(values=vals, weight=wts)
        new = op._descend(probe, cur, spec, alpha * probe.act_scale, n_steps, optimizer)
        if record is not None:
            with torch.no_grad():
                rec = record.setdefault(layer, {})
                rec["readout_err_before"] = float(
                    torch.sqrt((wts * (probe(cur) - vals) ** 2).sum(1).mean())
                )
                rec["readout_err_after"] = float(
                    torch.sqrt((wts * (probe(new) - vals) ** 2).sum(1).mean())
                )
                rec["delta_norm"] = float((new - cur).norm(dim=1).mean())
                rec["x_norm"] = float(cur.norm(dim=1).mean())
        out = x.clone()
        out[:, off:] = new.reshape(n, n_hist, x.shape[-1])
        return out

    tokens = model.embed(state.obs_buffer)
    h, _ = model._run(tokens, model._win_mask(state.length, dev), edit=hook)
    pred = model.decoder(model.norm_out(h[:, -1]))
    out = [pred]
    s = model.advance(state, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1).cpu().numpy()


def observation_history_edit(
    model,
    probe,
    obs_hist: np.ndarray,
    track: np.ndarray,
    delta: np.ndarray,
    edit_object: np.ndarray,
    *,
    alpha: float = 0.02,
    n_steps: int = 300,
    beta: float = 1.0,
    device: str = "cuda",
) -> tuple[np.ndarray, dict]:
    """The same MLP write applied to the **observations themselves** — no renderer.

    `probe` is the residual-point-0 probe, and residual point 0 is exactly
    `relu(Linear(obs))`, so `probe ∘ embed` is a differentiable map from an observation
    to a position read-out. Gradient descent on the observation through that composite
    changes the frames directly, which is the literal reading of "edit the previous
    observations", with the simulator never involved.
    """
    vals_np, wt_np = _history_targets(track, delta, edit_object, beta)
    n, n_hist, R = obs_hist.shape
    vals = torch.tensor(
        vals_np.reshape(n * n_hist, -1), device=device, dtype=torch.float32
    )
    wts = torch.tensor(
        wt_np.reshape(n * n_hist, -1), device=device, dtype=torch.float32
    )
    v = (
        torch.tensor(obs_hist, device=device, dtype=torch.float32)
        .clone()
        .requires_grad_(True)
    )

    with torch.no_grad():
        err0 = float(
            torch.sqrt(
                (wts * (probe(model.embed(v).reshape(n * n_hist, -1)) - vals) ** 2)
                .sum(1)
                .mean()
            )
        )

    opt = torch.optim.Adam([v], lr=alpha)
    for _ in range(n_steps):
        pred = probe(model.embed(v).reshape(n * n_hist, -1))
        loss = (wts * (pred - vals) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            v.clamp_(0.0, 1.0)  # observations are intensities in [0, 1]

    with torch.no_grad():
        edited = v.detach()
        err1 = float(
            torch.sqrt(
                (wts * (probe(model.embed(edited).reshape(n * n_hist, -1)) - vals) ** 2)
                .sum(1)
                .mean()
            )
        )
        d = float((edited - torch.tensor(obs_hist, device=device)).norm(dim=-1).mean())
        base = float(torch.tensor(obs_hist, device=device).norm(dim=-1).mean())
    return edited.cpu().numpy(), {
        "readout_err_before": err0,
        "readout_err_after": err1,
        "delta_norm": d,
        "x_norm": base,
    }
