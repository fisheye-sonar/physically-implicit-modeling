"""GS — Li et al. §4.1 MLP gradient steering, across layers. One of the three workhorses.

Moved verbatim 2026-08-31 from ``othello_gpt/othello_probe.py`` (the intervention half;
the probing half moved to ``pim.probes.base``). This is the port that reproduced their
published intervention on their own checkpoint (2026-08-20), used unmodified on every
model since — that provenance is the reason nothing here may be re-derived.

What is copied exactly from the paper:

* **The update rule** (§4.1): gradient descent on the *activation*, not the probe
  weights — ``x' ← x − α ∂L(p_θ(x), B') / ∂x``.
* **The sequential multi-layer schedule** (§4.1/Fig 2C): pick a starting layer L_s,
  intervene at the LAST timestep, let the network compute one more layer, intervene
  again, alternating to the final layer. A single-layer write is explicitly *not*
  their method — later layers recompute the stream from unaffected earlier positions
  and undo it.
* **The hold-the-rest term** (Appendix G): the loss carries the changed target plus
  ``β ×`` the error of every part that must stay put.

The probe the descent runs through is the canonical MLP (``pim.probes.mlp``); the
linear/regression adaptations are documented on ``EditSpec`` and ``_descend`` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pim.probes.base import WorldStateProbe

@dataclass
class EditSpec:
    """What the write must achieve, in the probe's own output space.

    values : (B, d_out) desired read-out. Dims being *changed* hold the intended new
             value; dims being *held* hold the probe's OWN pre-intervention reading —
             the paper's `B' = B except at tile s`, so the write injects no oracle
             information about anything but the target.
    weight : (B, d_out) 1.0 on changed dims, `beta` on held dims, 0 elsewhere.
    """

    values: torch.Tensor
    weight: torch.Tensor
    n_classes: int | None = None

    def loss(self, pred: torch.Tensor) -> torch.Tensor:
        if self.n_classes is None:
            return (self.weight * (pred - self.values) ** 2).sum(1).mean()
        # Classification: the paper's literal objective — per-tile cross-entropy against
        # `B' = B except at tile s`, weighted 1 on the changed tile and beta elsewhere,
        # then averaged (their `torch.mean(weight_mask * loss)`).
        ce = torch.nn.functional.cross_entropy(
            pred.reshape(-1, self.n_classes),
            self.values.reshape(-1).long(),
            reduction="none",
        ).reshape(self.values.shape)
        return (self.weight * ce).mean()


def build_edit_spec(
    probe: WorldStateProbe,
    x0: torch.Tensor,
    change_mask: np.ndarray | torch.Tensor,
    target_values: torch.Tensor,
    *,
    beta: float = 1.0,
) -> EditSpec:
    """Paper Appendix G loss, assembled around the probe's pre-intervention reading.

    change_mask   : (B, d_out) bool — which output dims the edit is trying to move.
    target_values : (B, d_out) desired values (only `change_mask` entries are read).
    beta          : weight on the hold-the-rest term.
    """
    with torch.no_grad():
        base = probe(x0)  # the paper's baseline state B
        if probe.n_classes is not None:
            base = base.argmax(-1)  # their `labels_pre_intv`
    cm = (
        change_mask
        if torch.is_tensor(change_mask)
        else torch.tensor(change_mask, device=x0.device)
    )
    cm = cm.bool()
    values = torch.where(cm, target_values.to(base.dtype), base)
    ones = torch.ones(base.shape, device=base.device, dtype=torch.float32)
    weight = torch.where(cm, ones, torch.full_like(ones, beta))
    return EditSpec(values=values, weight=weight, n_classes=probe.n_classes)


def _descend(
    probe: WorldStateProbe,
    x: torch.Tensor,
    spec: EditSpec,
    alpha: float,
    n_steps: int,
    optimizer: str = "adam",
) -> torch.Tensor:
    """The paper's §4.1 update `x' <- x - alpha * dL(p_theta(x), B')/dx`.

    `optimizer="gd"` is that rule literally. `optimizer="adam"` (the default) applies
    the same gradient through Adam, which the paper explicitly sanctions -- Appendix G:
    "this optimization process is robust to different configurations of optimizer,
    learning rate alpha, and number of steps". It is the default here because the
    residual points differ in scale by more than an order of magnitude (per-dim std
    medians 0.16 at point 0 to 2.76 at point 4), so no single raw-space alpha converges
    at every depth, and five hand-tuned alphas would be a worse-documented free
    parameter than one scale-invariant rule. Section 3b of the notebook verifies the two
    agree wherever plain GD is stable.
    """
    v = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        if optimizer == "adam":
            opt = torch.optim.Adam([v], lr=alpha)
            for _ in range(n_steps):
                loss = spec.loss(probe(v))
                opt.zero_grad()
                loss.backward()
                opt.step()
        elif optimizer == "gd":
            for _ in range(n_steps):
                v = v.detach().requires_grad_(True)
                loss = spec.loss(probe(v))
                (g,) = torch.autograd.grad(loss, v)
                v = v - alpha * g
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")
    return v.detach()


def make_intervention_hook(
    probes: dict[int, WorldStateProbe],
    specs: dict[int, EditSpec],
    start_layer: int,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    optimizer: str = "adam",
    record: dict | None = None,
):
    """The paper's Figure 2C schedule as a `_run` hook.

    Fires at **every** residual point; intervenes at the last timestep for every point
    ``>= start_layer``, letting the network recompute the stream in between. Points
    below ``start_layer`` pass through untouched.
    """

    def hook(layer: int, x: torch.Tensor) -> torch.Tensor:
        if layer < start_layer or layer not in probes:
            return x
        probe, spec = probes[layer], specs[layer]
        cur = x[:, -1]
        # `alpha` is relative to this residual point's activation scale, so one value
        # means the same size of write at every depth (see WorldStateProbe.act_scale).
        step = alpha * probe.act_scale
        new = _descend(probe, cur, spec, step, n_steps, optimizer)
        if record is not None:
            with torch.no_grad():
                rec = record.setdefault(layer, {})
                # The objective's own value, before and after. `hit_target` below is an
                # ARGMAX view of the same thing and saturates at 1.0 as soon as the write is
                # large enough to flip one label, which makes it useless for choosing a step
                # size once it does (measured 2026-08-20: 1.000 at every alpha in a 50x range).
                # The loss keeps improving after that, so it is the criterion that can still
                # tell a converged write from an under-converged one — and it is a property
                # of the WRITE, never of the outcome.
                rec["edit_loss_before"] = float(spec.loss(probe(cur)))
                rec["edit_loss_after"] = float(spec.loss(probe(new)))
                if spec.n_classes is None:
                    rec["readout_err_before"] = float(
                        torch.sqrt(
                            (spec.weight * (probe(cur) - spec.values) ** 2).sum(1).mean()
                        )
                    )
                    rec["readout_err_after"] = float(
                        torch.sqrt(
                            (spec.weight * (probe(new) - spec.values) ** 2).sum(1).mean()
                        )
                    )
                else:
                    # Classification analogue: the fraction of tiles the probe reads as the
                    # requested board B'. `hit_target` is the paper's own success criterion
                    # (`num_error == 0` on the intervened tile). DIAGNOSTIC ONLY — the write
                    # itself (`_descend`) is untouched by this branch.
                    tgt = spec.values.long()
                    chg = spec.weight >= 1.0
                    for tag, act in (("before", cur), ("after", new)):
                        hat = probe(act).argmax(-1)
                        rec[f"readout_err_{tag}"] = float((hat != tgt).float().mean())
                        rec[f"hit_target_{tag}"] = float(
                            (hat == tgt)[chg].float().mean()
                        )
                        rec[f"hold_rest_{tag}"] = float(
                            (hat == tgt)[~chg].float().mean()
                        )
                rec["delta_norm"] = float((new - cur).norm(dim=1).mean())
                rec["x_norm"] = float(cur.norm(dim=1).mean())
        out = x.clone()
        out[:, -1] = new
        return out

    return hook


@torch.no_grad()
def rollout_with_sequential_intervention(
    model,
    state,
    probes: dict[int, WorldStateProbe],
    specs: dict[int, EditSpec],
    start_layer: int,
    steps: int,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    optimizer: str = "adam",
    record: dict | None = None,
) -> torch.Tensor:
    """(B, steps, R) free-run whose FIRST step is produced under the intervention.

    Step 0 decodes the edit frame under the paper's multi-layer write; that prediction
    then enters the observation window and every later step is recomputed with **no**
    edit applied. Any persistence therefore has to travel through the observations,
    which is the property under test — and it is the same convention as
    ``TransformerModel.rollout_with_edit``, so these numbers sit on the same axis as
    every other editor in this thread.
    """
    hook = make_intervention_hook(
        probes,
        specs,
        start_layer,
        alpha=alpha,
        n_steps=n_steps,
        optimizer=optimizer,
        record=record,
    )
    tokens = model.embed(state.obs_buffer)
    dev = tokens.device
    h, _ = model._run(tokens, model._win_mask(state.length, dev), edit=hook)
    pred = model.decoder(model.norm_out(h[:, -1]))
    out = [pred]
    s = model.advance(state, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1)


@torch.no_grad()
def probe_readout_after_intervention(
    model,
    state,
    probes: dict[int, WorldStateProbe],
    specs: dict[int, EditSpec],
    start_layer: int,
    read_layer: int,
    *,
    alpha: float = 0.05,
    n_steps: int = 100,
    optimizer: str = "adam",
) -> torch.Tensor:
    """What `probes[read_layer]` reads once the intervention has been applied.

    Separates "did the optimisation land the read-out" from "did the generation follow"
    — the paper's own distinction, and the one that tells an inert write apart from a
    write the dynamics rejected.
    """
    hook = make_intervention_hook(
        probes, specs, start_layer, alpha=alpha, n_steps=n_steps, optimizer=optimizer
    )
    tokens = model.embed(state.obs_buffer)
    dev = tokens.device
    _, resids = model._run(
        tokens, model._win_mask(state.length, dev), edit=hook, want_resid=True
    )
    return probes[read_layer](resids[read_layer][:, -1])
