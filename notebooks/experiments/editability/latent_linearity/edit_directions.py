"""Latent edit-direction geometry — the one implementation for the `latent_linearity` thread.

Every *successful* edit mechanism in this repo gives us a ground-truth latent displacement

    Δh = h(model told about the edit) − h(matched control that was not)

This module builds those displacements for four mechanisms, on any architecture behind the
`HiddenStateModel` protocol, and measures whether they point the same way. Nothing here scores
observations — the canonical §4 metrics come from `scripts/editability_metrics.py`; nothing here
draws — figures come from `figures.py` beside it.

Design rules this file exists to enforce
----------------------------------------
1. **Native states, never round-tripped through a flat vector.** `state_from_flat` does not exist
   for a transformer's residual stream or a DiT's activations, and for the RSSM the round trip is
   worse than unavailable — it is *wrong*: `model.step` expects the **posterior** state at `t-1`,
   while the state aligned for `decode` is the **prior** at `t`. Feeding the prior back into `step`
   advances the deterministic core twice. `delta_h_analysis.ipynb`'s `continue_from` does exactly
   that, which is a live suspect for its anomalously weak RSSM freeze-time result (+0.09 vs the
   GRU's +0.54). Here the posterior chain is kept and `predictive()` is applied only at read-out.
2. **One alignment convention, stated once.** A *posterior* state has consumed frames `0..t`; its
   *predictive* form decodes frame `t+1`. Every analysed vector and every rollout starts from a
   predictive state, so `rollout[:, 0]` decodes the edit frame `ef` for every architecture.
3. **Mechanisms differ in how much world-time they consume**, so a Δh comparison has to say which
   frame the states are about. `Counterfactual Overwriting`, `Freeze-time Interp. TF` and
   `Action Interface` all leave the model about to predict `ef`; `First Obs. TF` has consumed the
   post-edit frame and is about to predict `ef+1`. `advance()` puts the first three at the second
   alignment so all four can be compared, and the notebook reports both alignments.

Metric definitions (formulas, units, better-direction) live in
`../METRICS_AND_EDITORS.md` §5. Import them from here; do not re-derive them in a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

# ── Model roster ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelSpec:
    """One (checkpoint, state object) pair to analyse.

    An architecture can expose more than one thing that deserves the name "state" — a
    transformer carries an observation buffer *and* recomputes a residual stream — so the unit of
    analysis is the pair, and `state_label` names which one every figure is about.
    """

    key: str  # short key used as a dict key
    label: str  # descriptive label for figures — never a bare run code
    ckpt: str  # path relative to the repo root
    arch: str  # architecture family
    state_label: str  # what the analysed state IS
    state_view: str | None = None  # runtime state-view toggle, where the model has one
    probe_layer: int | None = None  # residual point, for models that expose one
    run_code: str = ""  # registry row this checkpoint comes from
    registry: str = ""  # which run registry defines that row
    loader: str = "checkpoint"  # "checkpoint" | "action_sweep" (see `load_model`)


def load_model(spec: ModelSpec, root: str = "../../..", device: str = "cpu"):
    """Load a checkpoint and put it in the deterministic, analysis-ready configuration."""
    if spec.loader == "action_sweep":
        # The action-conditioned GRU is not reachable through `load_checkpoint`: its config has
        # the same keys as a plain GRU, so the dispatcher would silently build the wrong class
        # and drop the action channel. `eval_action_sweep.xg_load` is the one loader that reads
        # `use_actions` and picks the right one — import it rather than repeating the choice.
        from eval_action_sweep import xg_load

        model = xg_load(spec.run_code)[0].to(device)
    else:
        from pim.world_models import load_checkpoint

        model, _ = load_checkpoint(f"{root}/{spec.ckpt}", device=device)
    if hasattr(model, "sample"):
        model.sample = False  # RSSM: prior/posterior mean, never a draw
    if hasattr(model, "predict_mode"):
        model.predict_mode = (
            "mean"  # DiT family: the deterministic conditional-mean readout
        )
    if spec.state_view is not None:
        model.state_view = spec.state_view
    if spec.probe_layer is not None:
        model.probe_layer = spec.probe_layer
    return model


# ── State plumbing (architecture-agnostic) ────────────────────────────────────


def _is_rssm(model) -> bool:
    return hasattr(model, "imagine_step")


@torch.no_grad()
def observe(model, obs: torch.Tensor, upto: int, *, state=None, actions=None):
    """Teacher-force `obs[:, :upto]` and return the model's **posterior** state.

    The returned state has consumed frames `0..upto-1`; pass it to `predictive()` to get the
    state that decodes frame `upto`. Keeping the posterior is what lets `observe` be resumed
    (freeze-time appends frames to it) without the RSSM double-advance described in the module
    docstring.
    """
    for t in range(upto):
        if actions is None:
            _, state = model.step(obs[:, t], state)
        else:
            _, state = model.step(obs[:, t], state, action=actions[:, t])
    return state


@torch.no_grad()
def predictive(model, state):
    """Posterior state → the state whose `decode` gives the **next** frame.

    Identity for every architecture whose decoder already predicts next (GRU, transformer, DiT,
    latent DiT); one prior step for the RSSM, whose decoder reconstructs the current frame.
    """
    if _is_rssm(model):
        state, _ = model.imagine_step(state)
    return state


@torch.no_grad()
def advance(model, pred_state, steps: int = 1):
    """Free-run `steps` predictive states forward — no new evidence, the model's own dynamics.

    Used to put mechanisms that have not consumed the edit frame at the same alignment as
    `First Obs. TF`, which has.
    """
    for _ in range(steps):
        _, pred_state = model.predict_step(pred_state)
    return pred_state


@torch.no_grad()
def vec(model, pred_state) -> np.ndarray:
    """The analysed latent vector for a predictive state, `(N, H)` float64."""
    return model.flat_state(pred_state).double().cpu().numpy()


@torch.no_grad()
def rollout(model, pred_state, steps: int) -> np.ndarray:
    """Free-run `(N, steps, R)`; `[:, 0]` decodes the frame the predictive state is aligned to."""
    out = [model.decode(pred_state)]
    st = pred_state
    for _ in range(steps - 1):
        p, st = model.predict_step(st)
        out.append(p)
    return torch.stack(out, 1).float().cpu().numpy()


@torch.no_grad()
def step_norms(
    model, obs: torch.Tensor, *, upto: int, n_last: int = 5, actions=None
) -> np.ndarray:
    """‖h_t − h_{t−1}‖ averaged over the last `n_last` **pre-edit** transitions, per episode.

    The reference scale for "how big is this edit". Reported beside every Δh magnitude because a
    bare latent distance is uninterpretable and the scale differs by architecture
    (`harness/ANALYSIS.md` §7). Pre-edit only: the transition into the edit frame is the edit.
    """
    state, prev, diffs = None, None, []
    for t in range(upto):
        if actions is None:
            _, state = model.step(obs[:, t], state)
        else:
            _, state = model.step(obs[:, t], state, action=actions[:, t])
        cur = model.flat_state(predictive(model, state)).double()
        if prev is not None and t >= upto - n_last:
            diffs.append(torch.linalg.norm(cur - prev, dim=-1))
        prev = cur
    return torch.stack(diffs, 1).mean(1).cpu().numpy()


@torch.no_grad()
def alignment_profile(
    model,
    obs: torch.Tensor,
    clean_obs: np.ndarray,
    *,
    t_from: int = 12,
    t_to: int = 30,
    ks: tuple[int, ...] = (-2, -1, 0, 1),
) -> dict:
    """Which frame does `decode(predictive(state))` actually correspond to?

    Averaged over many `t` on **ordinary (non-edit) sequences** — on an edit sequence the
    pre-edit state legitimately fails to predict the edit frame, so the check would be confounded
    by the very effect under study.

    Read the profile **against `frame_change`**, the RMSE between consecutive clean frames. A
    predictor that hedges toward the frame it just saw can score lower at `k=−1` than at `k=0`
    while still being a correctly-aligned next-frame predictor — that is a statement about how
    far it moves, not about alignment. The decisive check is that the `k=0` value reproduces the
    model's independently published next-step RMSE against the clean render.
    """
    errs = {k: [] for k in ks}
    state = None
    for t in range(t_to + 1):
        _, state = model.step(obs[:, t], state)
        if t >= t_from:
            dec = model.decode(predictive(model, state)).float().cpu().numpy()
            for k in ks:
                tt = t + 1 + k
                if 0 <= tt < clean_obs.shape[1]:
                    errs[k].append(
                        float(np.sqrt(((dec - clean_obs[:, tt]) ** 2).mean()))
                    )
    prof = {k: float(np.mean(v)) for k, v in errs.items()}
    change = float(
        np.sqrt(
            (
                (clean_obs[:, t_from + 1 : t_to + 2] - clean_obs[:, t_from : t_to + 1])
                ** 2
            ).mean()
        )
    )
    return dict(
        profile=prof,
        argmin=int(min(prof, key=prof.get)),
        next_step_rmse_vs_clean=prof[0],
        frame_change=change,
    )


# ── The evidence each mechanism consumes ──────────────────────────────────────


@dataclass
class EditEvidence:
    """Rendered observation sequences that carry each mechanism's evidence for one edit set.

    Attributes
    ----------
    cf, cf_ctrl : (N, ef, R) counterfactual history in which the object always travelled to the
                  target, and the same renderer applied to the TRUE history. The control removes
                  everything the re-render changes except the edit itself.
    ft, ft_ctrl : (N, n_ft, R) freeze-time frames interpolating the object to the target with the
                  world otherwise frozen, and the same frames with the object held at its
                  pre-edit position.
    obs_ef, obs_ef_ctrl : (N, R) the single post-edit frame the model is shown by `First Obs. TF`,
                  and the same frame in the world where the edit never happened.
    """

    cf: np.ndarray
    cf_ctrl: np.ndarray
    ft: np.ndarray
    ft_ctrl: np.ndarray
    obs_ef: np.ndarray
    obs_ef_ctrl: np.ndarray
    pre_pos: np.ndarray
    tgt_pos: np.ndarray
    edit_object: np.ndarray


def build_evidence(
    *,
    positions: np.ndarray,  # (N, T, n_obj, 2) true positions
    velocities: np.ndarray,  # (N, T, n_obj, 2) true velocities
    edit_object: np.ndarray,  # (N,)
    target: np.ndarray,  # (N, 2) the edited object's post-edit position
    sim: dict,
    ef: int,
    n_obj: int = 2,
    n_ft: int = 8,
    uned_pos: np.ndarray | None = None,
    obs_noise_std: float | None = None,
    seed: int = 0,
) -> EditEvidence:
    """Render every mechanism's evidence, plus its matched no-edit control.

    Teacher-forced frames carry the dataset's observation noise (that is what the model consumed
    at training time); the clean render is reserved for scoring, per `GOTCHAS.md` 2026-08-04.
    """
    from pim.simulator.renderer import render_frame

    from editability_metrics import object_constants, sim_config_from

    cfg = sim_config_from(sim, n_obj)
    rad, refl = object_constants(sim, n_obj)
    dt = float(sim["dt"])
    R = int(sim["obs_res"])
    n = len(edit_object)
    noise = float(sim["obs_noise_std"]) if obs_noise_std is None else obs_noise_std
    rng = np.random.default_rng(seed)

    def render(pos_seq: np.ndarray) -> np.ndarray:
        """(F, n_obj, 2) → (F, R) clean intensities."""
        out = np.zeros((len(pos_seq), R), np.float32)
        for j, p in enumerate(pos_seq):
            out[j] = render_frame(p.astype(np.float32), rad, refl, cfg)[2]
        return out

    ix = np.arange(n)
    obj = edit_object.astype(int)
    other = 1 - obj
    tgt_pos = positions[:, ef].copy()
    tgt_pos[ix, obj] = target
    pre_pos = positions[:, ef - 1]

    # The world where the edit never happened. On an edits split whose teleport is *in the data*,
    # `positions[ef]` is already the post-edit world, so the counterfactual has to be constructed
    # ballistically from `ef-1`. On an intervention-free eval set where the teleport is
    # *synthesised*, `positions[ef]` IS the un-teleported world and should be passed in — it is
    # exact, where the ballistic estimate differs from it by the world's position noise.
    if uned_pos is None:
        uned_pos = tgt_pos.copy()
        uned_pos[ix, obj] = pre_pos[ix, obj] + velocities[ix, ef - 1, obj] * dt

    cf = np.zeros((n, ef, R), np.float32)
    cf_ctrl = np.zeros((n, ef, R), np.float32)
    ft = np.zeros((n, n_ft, R), np.float32)
    ft_ctrl = np.zeros((n, n_ft, R), np.float32)
    obs_ef = np.zeros((n, R), np.float32)
    obs_ef_ctrl = np.zeros((n, R), np.float32)
    t_idx = np.arange(ef)

    for i in range(n):
        o, oth = obj[i], other[i]
        v = velocities[i, ef, o]
        # counterfactual history: a constant-velocity line that ARRIVES at the target at `ef`
        hist = np.zeros((ef, n_obj, 2), np.float32)
        hist[:, o] = tgt_pos[i, o][None] - v[None] * (ef - t_idx)[:, None] * dt
        hist[:, oth] = positions[i, :ef, oth]
        cf[i] = render(hist)
        cf_ctrl[i] = render(positions[i, :ef])
        # freeze-time: interpolate pre → target with the world otherwise frozen at frame `ef`
        fr = np.zeros((n_ft, n_obj, 2), np.float32)
        fc = np.zeros((n_ft, n_obj, 2), np.float32)
        for j in range(n_ft):
            fr[j, o] = pre_pos[i, o] + ((j + 1) / n_ft) * (
                tgt_pos[i, o] - pre_pos[i, o]
            )
            fc[j, o] = pre_pos[i, o]
            fr[j, oth] = fc[j, oth] = tgt_pos[i, oth]
        ft[i] = render(fr)
        ft_ctrl[i] = render(fc)
        obs_ef[i] = render(tgt_pos[i][None])[0]
        obs_ef_ctrl[i] = render(uned_pos[i][None])[0]

    # Noise-match every teacher-forced frame to what the model consumed in training, and give a
    # mechanism's edited and control arms the **same** noise draw. Independent draws would inject
    # a difference of order 2·0.2 per ray over 20 frames into every Δh — unrelated to the edit,
    # and large enough to dominate the cosines this module exists to measure.
    for edited, ctrl in ((cf, cf_ctrl), (ft, ft_ctrl), (obs_ef, obs_ef_ctrl)):
        eps = rng.normal(0.0, noise, edited.shape).astype(np.float32)
        edited += eps
        ctrl += eps

    return EditEvidence(
        cf=cf,
        cf_ctrl=cf_ctrl,
        ft=ft,
        ft_ctrl=ft_ctrl,
        obs_ef=obs_ef,
        obs_ef_ctrl=obs_ef_ctrl,
        pre_pos=pre_pos,
        tgt_pos=tgt_pos,
        edit_object=obj,
    )


# ── The four mechanisms, as (edited state, matched control state) pairs ───────

#: Canonical mechanism names — `../METRICS_AND_EDITORS.md`. Use these, and only these, in figures.
COUNTERFACTUAL = "Counterfactual Overwriting"
FREEZE_TIME = "Freeze-time Interp. TF @8"
ACTION = "Action Interface"
FIRST_OBS = "First Obs. TF"

#: How many frames of world-time each mechanism has consumed past the edit frame.
CONSUMES_EDIT_FRAME = {
    COUNTERFACTUAL: False,
    FREEZE_TIME: False,
    ACTION: False,
    FIRST_OBS: True,
}


@dataclass
class MechanismStates:
    """Predictive states for one mechanism: the edited arm and its matched control."""

    edited: Any
    control: Any
    consumes_edit_frame: bool


@dataclass
class ModelRun:
    """Everything measured for one (model, state object): states, vectors, references."""

    spec: ModelSpec
    unsteered: Any  # predictive state, no edit
    mechanisms: dict[str, MechanismStates] = field(default_factory=dict)
    step_norm: np.ndarray | None = None  # (N,) one-dynamics-step reference scale
    h0_norm: np.ndarray | None = None  # (N,) ‖unsteered state‖


@torch.no_grad()
def build_states(
    model,
    spec: ModelSpec,
    ev: EditEvidence,
    obs: np.ndarray,
    ef: int,
    *,
    device: str = "cpu",
    actions_edit: np.ndarray | None = None,
    actions_noop: np.ndarray | None = None,
) -> ModelRun:
    """Build the unsteered state and every applicable mechanism's (edited, control) pair.

    `actions_edit` / `actions_noop` are supplied only for a model whose action channel contains
    the intervention; `Action Interface` is skipped for every other model, which is a structural
    absence, not a missing measurement.
    """
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)  # noqa: E731
    o = t(obs)
    a_edit = None if actions_edit is None else t(actions_edit)
    a_noop = None if actions_noop is None else t(actions_noop)

    post0 = observe(model, o, ef, actions=a_noop)
    pred0 = predictive(model, post0)
    run = ModelRun(spec=spec, unsteered=pred0)

    # 1 — counterfactual overwrite: a whole rewritten history, teacher-forced from scratch
    run.mechanisms[COUNTERFACTUAL] = MechanismStates(
        edited=predictive(model, observe(model, t(ev.cf), ef, actions=a_noop)),
        control=predictive(model, observe(model, t(ev.cf_ctrl), ef, actions=a_noop)),
        consumes_edit_frame=False,
    )

    # 2 — freeze-time: extra frames appended to the real history, world otherwise frozen
    n_ft = ev.ft.shape[1]
    run.mechanisms[FREEZE_TIME] = MechanismStates(
        edited=predictive(model, observe(model, t(ev.ft), n_ft, state=post0)),
        control=predictive(model, observe(model, t(ev.ft_ctrl), n_ft, state=post0)),
        consumes_edit_frame=False,
    )

    # 3 — action interface: the same observations, the teleport issued through the action channel
    if a_edit is not None:
        run.mechanisms[ACTION] = MechanismStates(
            edited=predictive(model, observe(model, o, ef, actions=a_edit)),
            control=pred0,
            consumes_edit_frame=False,
        )

    # 4 — first observation: one post-edit frame teacher-forced onto the real history
    run.mechanisms[FIRST_OBS] = MechanismStates(
        edited=predictive(model, observe(model, t(ev.obs_ef)[:, None], 1, state=post0)),
        control=predictive(
            model, observe(model, t(ev.obs_ef_ctrl)[:, None], 1, state=post0)
        ),
        consumes_edit_frame=True,
    )

    run.step_norm = step_norms(model, o, upto=ef, actions=a_noop)
    run.h0_norm = np.linalg.norm(vec(model, pred0), axis=-1)
    return run


def deltas(
    model,
    run: ModelRun,
    *,
    align_to_edit_frame: bool = True,
) -> dict[str, np.ndarray]:
    """Δh per mechanism, `edited − control`, all at one alignment.

    `align_to_edit_frame=True` measures every mechanism at "about to predict `ef`", which means
    `First Obs. TF` is excluded — it has already consumed that frame and cannot be un-advanced.
    `False` measures every mechanism at "about to predict `ef+1`", free-running the others one
    step so all four are comparable. Both are reported; neither is allowed to be implicit.
    """
    out: dict[str, np.ndarray] = {}
    for name, m in run.mechanisms.items():
        if align_to_edit_frame:
            if m.consumes_edit_frame:
                continue
            e, c = m.edited, m.control
        else:
            n = 0 if m.consumes_edit_frame else 1
            e, c = advance(model, m.edited, n), advance(model, m.control, n)
        out[name] = vec(model, e) - vec(model, c)
    return out


# ── Geometry ──────────────────────────────────────────────────────────────────


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point, so a shuffled pair never contains a true pair."""
    for _ in range(100):
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p
    p = np.roll(np.arange(n), 1)
    return p


def cosine_report(u: np.ndarray, v: np.ndarray, *, seed: int = 0) -> dict:
    """Do two edit directions point the same way? Per instance, then averaged.

    Returns the signed cosine (chance level 0 at every `H`), the angle, the **projection
    fraction** `mean|cos|` — the share of one displacement's magnitude that lies along the other —
    and both against a **shuffled-pair** control, which is the empirical chance level and the
    stricter test than any formula. `enrichment` is the projection fraction over its shuffled
    control and is the quantity to compare **across architectures**, because `mean|cos|` for
    unrelated vectors falls as `√(2/πH)` and `H` differs by model (`harness/ANALYSIS.md` §8).
    """
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    n, H = u.shape
    cos = (_unit(u) * _unit(v)).sum(-1)
    rng = np.random.default_rng(seed)
    perm = _derangement(n, rng)
    cos_s = (_unit(u) * _unit(v[perm])).sum(-1)
    mean = float(cos.mean())
    return dict(
        n=n,
        H=H,
        cos_mean=mean,
        cos_sd=float(cos.std()),
        cos_median=float(np.median(cos)),
        angle_deg=float(np.degrees(np.arccos(np.clip(mean, -1, 1)))),
        proj_frac=float(np.abs(cos).mean()),
        cos_shuffled=float(cos_s.mean()),
        cos_shuffled_sd=float(cos_s.std()),
        proj_frac_shuffled=float(np.abs(cos_s).mean()),
        enrichment=float(np.abs(cos).mean() / max(np.abs(cos_s).mean(), 1e-12)),
        z_vs_shuffled=float((mean - cos_s.mean()) / max(cos_s.std(), 1e-12)),
        cos_per_sample=cos,
        cos_shuffled_per_sample=cos_s,
    )


def magnitude_report(
    dh: np.ndarray, h0_norm: np.ndarray, step_norm: np.ndarray
) -> dict:
    """How big is the edit, in the two reference scales that make a latent distance readable.

    `rel_state` = ‖Δh‖ / ‖h‖ (is the edit the size of the state?) and `rel_step` = ‖Δh‖ / one
    ordinary dynamics step (is it the size of something the dynamics does anyway?). The second is
    the one to compare across architectures — ‖h‖ has no common meaning across state objects.
    """
    norm = np.linalg.norm(np.asarray(dh, np.float64), axis=-1)
    rel_state = norm / np.clip(h0_norm, 1e-12, None)
    rel_step = norm / np.clip(step_norm, 1e-12, None)
    return dict(
        norm_mean=float(norm.mean()),
        rel_state_mean=float(rel_state.mean()),
        rel_state_sd=float(rel_state.std()),
        rel_step_mean=float(rel_step.mean()),
        rel_step_sd=float(rel_step.std()),
        cv=float(norm.std() / max(norm.mean(), 1e-12)),
        norm_per_sample=norm,
        rel_step_per_sample=rel_step,
    )


def consistency_report(dh: np.ndarray, *, n_pairs: int = 20000, seed: int = 0) -> dict:
    """Is there ONE latent direction for "an object moved", shared across different edits?

    Mean cosine between the Δh of *different* edit samples. For unrelated directions the expected
    mean is **0** — `1/√H` is the per-pair standard deviation, not a floor (`ANALYSIS.md` §8.2) —
    so the sd is reported beside it and the mean is read against 0.
    """
    dh = _unit(np.asarray(dh, np.float64))
    n = len(dh)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    cos = (dh[i[keep]] * dh[j[keep]]).sum(-1)
    return dict(
        pairwise_cos_mean=float(cos.mean()),
        pairwise_cos_sd=float(cos.std()),
        sem=float(cos.std() / np.sqrt(len(cos))),
        n_pairs=int(keep.sum()),
    )


def rowspace_report(dh: np.ndarray, A: np.ndarray) -> dict:
    """How much of Δh can a linear position probe even see?

    `f = ‖P_row·Δh‖ / ‖Δh‖` with `P_row = A⁺A` the projector onto the probe's row space — also
    the ceiling on the cosine any injection-style write could reach with this Δh. A random vector
    already scores `√(d/H)`, so **enrichment over that chance level** is what is comparable across
    architectures; `f` alone manufactures a trend that is entirely the moving chance level.
    """
    dh = np.asarray(dh, np.float64)
    A = np.asarray(A, np.float64)
    d, H = A.shape
    P_row = np.linalg.pinv(A) @ A
    f = np.linalg.norm(dh @ P_row.T, axis=-1) / np.clip(
        np.linalg.norm(dh, axis=-1), 1e-12, None
    )
    chance = float(np.sqrt(d / H))
    return dict(
        f_mean=float(f.mean()),
        f_sd=float(f.std()),
        chance=chance,
        enrichment=float(f.mean() / chance),
        d=int(d),
        H=int(H),
    )


def fit_position_probe(
    model,
    obs: np.ndarray,
    positions: np.ndarray,
    *,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """The standard linear+MLP read-out probe, fit on **predictive** states.

    Wraps `pim.extractors.fit_readability_probes` (the one standard estimator: linear lstsq plus a
    2×256 MLP, both fit on the same 80% of *sequences* and scored on the held-out 20%) and takes
    its fitted `A` for the row-space projector. The states are the same *kind* the edits are
    measured on — decode(state) ↔ frame t+1 — so the probe is not applied one frame off the
    states it was fit on, which for the RSSM is a real mismatch rather than a nicety.
    """
    from pim.extractors import fit_readability_probes

    o = torch.from_numpy(np.asarray(obs, np.float32)).to(device)
    T = o.shape[1]
    bank, state = [], None
    with torch.no_grad():
        for t in range(T - 1):
            _, state = model.step(o[:, t], state)
            bank.append(model.flat_state(predictive(model, state)))
    states = torch.stack(bank, 1).float().cpu().numpy()
    y = positions[:, 1 : 1 + states.shape[1]].reshape(len(states), states.shape[1], -1)
    out = fit_readability_probes(states, y, device=device, seed=seed)
    return out


# ── Assembly helpers used by the notebook ─────────────────────────────────────


def alignment_matrix(dh: dict[str, np.ndarray], *, seed: int = 0) -> dict:
    """Pairwise `cosine_report` between every mechanism present, as a dict of dicts."""
    names = list(dh)
    return {
        a: {b: cosine_report(dh[a], dh[b], seed=seed) for b in names if b != a}
        for a in names
    }


def as_matrix(mat: dict, names: list[str], field_name: str = "cos_mean") -> np.ndarray:
    """`alignment_matrix` → a dense array for plotting; the diagonal is 1 by definition."""
    out = np.eye(len(names))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if a != b and b in mat.get(a, {}):
                out[i, j] = mat[a][b][field_name]
    return out
