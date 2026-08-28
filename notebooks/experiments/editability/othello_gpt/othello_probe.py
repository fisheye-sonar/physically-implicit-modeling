"""Paper-exact probes and layer-wise intervention, after Li et al. (ICLR 2023).

*Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic
Task* (arXiv:2210.13382). This module ports their §3 probing and §4 intervention
methodology onto this repo's causal transformer world model, changing only what the
different task forces.

What is copied exactly
----------------------
* **The probe family.** Paper §3.2: a 2-layer MLP,
  ``p_θ(x) = softmax(W₁ ReLU(W₂ x))`` — i.e. ONE hidden layer — compared against the
  §3.1 linear probe ``p_θ(x) = softmax(W x)``. Both are fit per residual point, and
  accuracy is reported across depth (their Tables 1 and 2).
* **The intervention rule.** Paper §4.1: gradient descent on the *activation*, not the
  probe weights, ``x' ← x − α ∂L(p_θ(x), B') / ∂x``.
* **The sequential multi-layer schedule.** Paper §4.1/Figure 2C: pick a starting layer
  ``L_s``, intervene at the **last timestep**, let the network compute one more layer,
  intervene again, and alternate to the final layer. Intervening at a single layer is
  explicitly *not* what they do — later layers recompute the stream from unaffected
  earlier positions and undo it.
* **The hold-the-rest term.** Paper Appendix G: the loss carries the changed target plus
  ``β ×`` the cross-entropy of every part that must stay put, so the write moves one
  thing and leaves the rest alone.
* **The baseline they measure against.** Paper §4.2's *null intervention* — score the
  pre-intervention prediction against the post-intervention ground truth.

What necessarily differs, and why
---------------------------------
* **Regression, not 3-way classification.** Othello's board state is 64 ternary tiles;
  this world's state is continuous object positions (and velocities). So the softmax and
  cross-entropy become a linear read-out and squared error, and probe quality is reported
  as **R²** rather than a classification error rate. The probe *shape* is unchanged.
* **Held-out split is by SEQUENCE, not by frame.** The paper splits (activation, tile)
  pairs 8:2 at random, which for this world is a leak: velocity is constant along a
  trajectory, so a frame-level split leaves the identical label in train for every test
  frame (measured inflation +0.34 R² on velocity — ``research/GOTCHAS.md``, 2026-08-14).
  This is a deliberate, documented deviation; matching the paper here would inflate every
  number in the accuracy-across-layers figure.
* **Standardisation is inside the probe.** Fitting these targets without standardising
  diverges (``research/GOTCHAS.md``). The probe stores the moments as buffers and takes a
  **raw** activation, so ``∂L/∂x`` is taken in raw activation space exactly as the paper
  writes it, while the optimiser sees a well-scaled problem.

This is a *different object* from ``pim.extractors.fit_readability_probes`` (2 hidden
layers × 256, the repo's readability standard). Numbers from the two are not comparable
and must never be quoted as one another — see ``harness/ANALYSIS.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ── Probes ────────────────────────────────────────────────────────────────────


class WorldStateProbe(nn.Module):
    """Paper §3 probe: linear, or a 2-layer MLP with ONE hidden layer.

    Two output modes, same body:

    * ``n_classes=None`` (default) — **regression**. Maps a **raw** residual-stream
      activation to world-state values in **raw sim units**; ``d_out`` is the number of
      scalars. This is what runs on this repo's own world model.
    * ``n_classes=C`` — **C-way classification over ``d_out`` tiles**, added 2026-08-20
      for `othello_transfer/`, where the state is 64 ternary tiles rather than continuous
      scalars. ``forward`` then returns ``(B, d_out, C)`` logits and the ``y`` affine is
      skipped (it is meaningless on logits). The body is otherwise **identical**, and at
      ``C=3, d_out=64`` it is exactly the shape of Li et al.'s own
      ``BatteryProbeClassificationTwoLayer``.

    Standardisation moments are non-trainable buffers either way, so the module is a
    pure function of the raw activation and ``∂L/∂x`` is taken in the space the
    paper's update rule assumes.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden: int | None = 512,
        *,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.d_in, self.d_out, self.hidden = d_in, d_out, hidden
        self.n_classes = n_classes
        n_final = d_out if n_classes is None else d_out * n_classes
        if hidden is None:  # paper §3.1 — linear probe
            self.net = nn.Linear(d_in, n_final)
        else:  # paper §3.2 — softmax(W1 ReLU(W2 x)), one hidden layer
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, n_final)
            )
        z = torch.zeros(d_in)
        o = torch.ones(d_in)
        self.register_buffer("x_mean", z.clone() if x_mean is None else x_mean)
        self.register_buffer("x_std", o.clone() if x_std is None else x_std)
        zo, oo = torch.zeros(d_out), torch.ones(d_out)
        self.register_buffer("y_mean", zo.clone() if y_mean is None else y_mean)
        self.register_buffer("y_std", oo.clone() if y_std is None else y_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, d_in) raw activation → (B, d_out) values, or (B, d_out, n_classes) logits."""
        z = (x - self.x_mean) / self.x_std
        out = self.net(z)
        if self.n_classes is not None:
            return out.reshape(*out.shape[:-1], self.d_out, self.n_classes)
        return out * self.y_std + self.y_mean

    @property
    def kind(self) -> str:
        base = "linear" if self.hidden is None else f"mlp-{self.hidden}"
        return base if self.n_classes is None else f"{base}-{self.n_classes}way"

    @property
    def act_scale(self) -> float:
        """Typical per-dim spread of the activations this probe was fit on.

        The residual points differ in scale by ~17x (median per-dim std 0.16 at point 0
        to 2.76 at point 4), so a single absolute step size cannot converge at every
        depth. Intervention step sizes are quoted **relative** to this, which makes one
        `alpha` mean the same thing at every residual point.
        """
        return float(self.x_std.median())


# ── Residual collection ───────────────────────────────────────────────────────


@torch.no_grad()
def collect_residuals(model, obs: np.ndarray, batch: int = 128) -> np.ndarray:
    """(n_points, N, T, d_model) residual stream at every residual point/position.

    Uses the model's one-pass banded forward, which *is* the sliding window (the band
    mask is the window), so this is exactly the activation a step-by-step rollout
    would produce at each position — verified by
    ``tests/test_transformer.py::test_buffer_rollout_matches_full_sequence``.
    """
    dev = next(model.parameters()).device
    out: list[np.ndarray] = []
    for i in range(0, len(obs), batch):
        o = torch.from_numpy(obs[i : i + batch]).float().to(dev)
        tokens = model.embed(o)
        _, resids = model._run(
            tokens, model._seq_mask(o.shape[1], dev), want_resid=True
        )
        out.append(torch.stack(resids, 0).float().cpu().numpy())
    return np.concatenate(out, axis=1)


# ── Fitting ───────────────────────────────────────────────────────────────────


def _r2(pred: np.ndarray, gt: np.ndarray, train_mean: np.ndarray) -> float:
    """R² against the TRAIN mean (repo convention), pooled over output dims."""
    ss_res = ((pred - gt) ** 2).sum()
    ss_tot = ((gt - train_mean) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def fit_probe(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    hidden: int | None = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    batch: int = 4096,
    device: str = "cuda",
    seed: int = 0,
    n_classes: int | None = None,
) -> tuple[WorldStateProbe, dict]:
    """Fit one probe. `hidden=None` gives the paper's linear probe.

    ``n_classes=None`` (default) is the regression fit used on this repo's own world
    model: squared error in standardised target space, and the linear probe solved in
    closed form.

    ``n_classes=C`` (added 2026-08-20) is the **classification** fit for a state made of
    ``d_out`` C-way tiles, which is what Li et al. actually train. ``y`` then holds
    integer class labels ``(N, d_out)``, the loss is per-tile cross-entropy, and quality
    is reported as an **error rate (%)** so it sits beside their Tables 1 and 2. The
    linear probe is trained by the same SGD loop rather than ``lstsq``, which has no
    multinomial analogue — so unlike the regression path, the linear-vs-MLP gap here
    carries an optimiser difference of zero rather than an optimiser *advantage* to the
    linear side.
    """
    torch.manual_seed(seed)
    xm, xs = x_tr.mean(0), x_tr.std(0)
    # Floor the per-dim scale. Residual point 0 spans a 600x range of per-dim std
    # (2.6e-4 against a 0.16 median): dividing by the smallest makes the standardised
    # objective so anisotropic that activation-space gradient descent diverges at any
    # step size large enough to move the read-out. The floor costs nothing in fit
    # quality (the probe compensates with larger weights) and makes the descent
    # well-conditioned, which is what the paper's update rule assumes.
    xs = np.maximum(xs, 1e-2 * np.median(xs)) + 1e-8
    if n_classes is None:
        ym, ys = y_tr.mean(0), y_tr.std(0) + 1e-6
    else:  # logits carry no target affine
        ym = np.zeros(y_tr.shape[1], np.float32)
        ys = np.ones(y_tr.shape[1], np.float32)

    probe = WorldStateProbe(
        x_tr.shape[1],
        y_tr.shape[1],
        hidden,
        x_mean=torch.tensor(xm, dtype=torch.float32),
        x_std=torch.tensor(xs, dtype=torch.float32),
        y_mean=torch.tensor(ym, dtype=torch.float32),
        y_std=torch.tensor(ys, dtype=torch.float32),
        n_classes=n_classes,
    ).to(device)

    if n_classes is not None:
        # Classification path: the paper's own objective, per-tile cross-entropy.
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.long, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                logits = probe(xt[idx])  # (b, d_out, C)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_classes), yt[idx].reshape(-1)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        probe.eval()

        def _pred(x):
            out = []
            with torch.no_grad():
                for i in range(0, len(x), 8192):
                    xb = torch.tensor(x[i : i + 8192], dtype=torch.float32, device=device)
                    out.append(probe(xb).argmax(-1).cpu().numpy())
            return np.concatenate(out, 0)

        hat_te, hat_tr = _pred(x_te), _pred(x_tr)
        per_tile_err = (hat_te != y_te).mean(0)
        stats = {
            "error_rate": float((hat_te != y_te).mean() * 100.0),
            "error_rate_insample": float((hat_tr != y_tr).mean() * 100.0),
            "accuracy": float((hat_te == y_te).mean() * 100.0),
            "per_tile_error_rate": (per_tile_err * 100.0).tolist(),
            "majority_class_error_rate": float(
                (1.0 - np.bincount(y_tr.reshape(-1), minlength=n_classes).max()
                 / y_tr.size) * 100.0
            ),
            "n_train_rows": int(len(x_tr)),
            "n_test_rows": int(len(x_te)),
            "kind": probe.kind,
        }
        return probe, stats

    if hidden is None:
        # Closed-form least squares: strictly better than SGD for a linear map, so the
        # linear-vs-MLP gap is a statement about the probe family, not the optimiser.
        zx = (x_tr - xm) / xs
        zy = (y_tr - ym) / ys
        A = np.concatenate([zx, np.ones((len(zx), 1), dtype=np.float32)], 1)
        W, *_ = np.linalg.lstsq(A, zy, rcond=None)
        with torch.no_grad():
            probe.net.weight.copy_(torch.tensor(W[:-1].T, dtype=torch.float32))
            probe.net.bias.copy_(torch.tensor(W[-1], dtype=torch.float32))
    else:
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        opt = torch.optim.Adam(probe.parameters(), lr=lr)
        n = len(xt)
        # The loss is taken in STANDARDISED target space, so every output dimension
        # contributes equally.
        #
        # `forward` un-standardises (`net(z) * y_std + y_mean`), so a raw-units loss is
        # implicitly weighted by `y_std**2` per dimension — and on a combined
        # position+velocity target that is a ~1000x weighting toward position (variance
        # 3.0-3.6 vs 0.0033 in sim units). The velocity dimensions were then barely
        # trained: measured 2026-08-19, mean velocity R2 0.158 with the raw-units loss
        # vs 0.276 here, with the x-components roughly doubling (0.211 -> 0.518,
        # 0.450 -> 0.730) and position paying almost nothing (0.938 -> 0.927). The
        # balanced fit also matches a dedicated velocity-only probe (0.272), so the
        # imbalance was the whole gap.
        #
        # The tell was that the MLP scored BELOW the linear probe on velocity
        # (0.158 vs 0.200). A strictly more expressive probe losing to a linear one is a
        # training failure, never a fact about the representation - the same diagnostic
        # the repo standard records for its own 2026-08-11 undertraining fix.
        ys_t = probe.y_std.detach()
        for ep in range(epochs):
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                loss = (((probe(xt[idx]) - yt[idx]) / ys_t) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    probe.eval()
    with torch.no_grad():
        pr_te = (
            probe(torch.tensor(x_te, dtype=torch.float32, device=device)).cpu().numpy()
        )
        pr_tr = (
            probe(torch.tensor(x_tr, dtype=torch.float32, device=device)).cpu().numpy()
        )
    stats = {
        "r2": _r2(pr_te, y_te, ym),
        "r2_insample": _r2(pr_tr, y_tr, ym),
        "rmse": float(np.sqrt(((pr_te - y_te) ** 2).mean())),
        "per_dim_r2": [
            _r2(pr_te[:, [j]], y_te[:, [j]], ym[[j]]) for j in range(y_te.shape[1])
        ],
        "kind": probe.kind,
    }
    return probe, stats


# ── Paper §4.1 intervention ───────────────────────────────────────────────────


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
