"""Edit waterfalls on the dw-blink REAPPEARANCE cases, with the blink markers visible.

Why this exists (2026-09-08). The canonical panel (`notebooks/make_waterfalls.ipynb`,
`runs/blink_ablation/L-dw-blink-20m/figures/waterfall_edits.png`) draws the first 192 edit
cases, which are ~79% visible / ~18% mid-blackout / ~3% reappearance, so it shows the blink
result at whatever phase the case happens to be in. It also hides the 0.5 blackout markers:
they occupy ray 0 and ray 127 of a 128-wide strip drawn ~120 px wide, so they fall under the
axis spine. Both are figure problems, not data problems (the markers are stored and are
reconstructed into `clean_obs`; no sequence is hidden before frame 3).

This script fixes both:
  * cases are SELECTED by blink phase — the reappearance subset (edited object hidden
    through EF-1, visible at EF, staleness k >= --min-k), the same selection rule as
    `experiments/blink_ablation/scripts/subset_editability.py`, in index order (no
    cherry-picking) — and, for contrast, the visible control under the identical arms;
  * the two EDGE RAYS ARE WIDENED to `--edge-width` pixels in every column, so a marker is
    visible, and the widened track is fenced off with a thin rule. The widening is a
    DRAWING transform applied identically to GT, the unsteered run and every editor, so a
    marker the model failed to predict stays absent.
  * the context window is deeper than the canonical 6 frames (default 10) so the blackout
    START marker is usually on screen as well as the END marker at EF-1.

Everything else is canonical: rollouts from `pim.environments.discworld.arms`, zones and
Edit Index from `pim.metrics`, drawing through `pim.figures.waterfall_grid` per
`research/specs/WATERFALL_SPEC.md`. Arms come from the subset analysis's own scores
(`experiments/blink_ablation/scores/subset_editability.json`) so the picture is drawn at
exactly the arm the subset table reports; `--guard` instead picks that subset's best arm
with fidelity ratio <= the guard, which is the honest arm to LOOK at (the raw best PI arm
on this model is a destructive alpha=175 write).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ
from pim.environments.discworld import blink as bk
from pim.figures import waterfall_grid
from pim.metrics.edit_index import edit_index_per_case
from pim.models import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/blink_ablation/L-dw-blink-20m")
ap.add_argument("--subsets", nargs="+", default=["reappearance", "visible"])
ap.add_argument("--n", type=int, default=6, help="waterfall rows per subset")
ap.add_argument("--min-k", type=int, default=3, help="staleness floor for reappearance cases")
ap.add_argument("--basis", default="frustum")
ap.add_argument("--n-ctx", type=int, default=10, help="context frames drawn above the line")
ap.add_argument("--edge-width", type=int, default=4, help="drawn width of each edge (marker) ray")
ap.add_argument("--guard", type=float, default=1.1,
                help="draw each editor's best arm with fidelity ratio <= this (0 = raw best)")
ap.add_argument("--probe-seqs", type=int, default=30_000)
ap.add_argument("--out", default="experiments/blink_ablation/waterfalls/outputs")
a = ap.parse_args()

run_dir = REPO / a.run
cfg = json.loads((run_dir / "config.json").read_text())
inst = cfg["data"]["instance"]
inst_root = REPO / "datasets" / "discworld" / inst
eval_dir, probe_corpus = inst_root / "eval", inst_root / "probe"
out_dir = REPO / a.out
out_dir.mkdir(parents=True, exist_ok=True)

model, info = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
print(f"{a.run} · {info.arch} · val {info.val_loss:.6f} · instance {inst}", flush=True)

# ── the blink phase of every edit case (the subset_editability.py rule) ──────────────
with h5py.File(eval_dir / "edits.h5", "r") as f:
    vis = f["blink_visible"][:, :, :N_OBJ].astype(bool)
    eobj = f["edit_object"][:].astype(int)
    assert int(f["edit_frame"][0]) == EF
M = len(eobj)
vis_e = vis[np.arange(M), :, eobj]                       # (M, T) the EDITED object
k = np.zeros(M, int)                                     # hidden frames immediately before EF
for i in range(M):
    t = EF - 1
    while t >= 0 and not vis_e[i, t]:
        k[i] += 1
        t -= 1
SUBSETS = {
    "reappearance": np.where((k >= a.min_k) & vis_e[:, EF])[0],
    "mid_blackout": np.where(~vis_e[:, EF])[0],
    "visible": np.where(vis_e[:, EF - 1] & vis_e[:, EF])[0],
}

# ── probes (cached in the run dir) and the arms the subset table reports ─────────────
lin = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="linear",
                     basis_name=a.basis, data_dir=probe_corpus,
                     cache_dir=run_dir / "probes", log=None)
mlp = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="mlp",
                     basis_name=a.basis, data_dir=probe_corpus,
                     cache_dir=run_dir / "probes", log=None)
SUB = json.loads((REPO / "experiments/blink_ablation/scores/subset_editability.json")
                 .read_text())["bases"][a.basis]["subsets"]

def arms_for(subset: str) -> dict:
    """{editor: (point, alpha, dims, fidelity, subset EI)} — that subset's best arm,
    optionally fidelity-guarded. The subset EI is the 192-case number the table reports;
    it travels with the arm so the figure's own few rows can never be read as the result."""
    key = "ei_at_reappearance" if subset == "mid_blackout" else "edit_index"
    src = SUB["reappearance" if subset == "reappearance" else subset]
    out = {}
    for ed in ("PI", "GS"):
        recs = [r for r in src["arms"] if r["editor"] == ed and np.isfinite(r[key])
                and (a.guard <= 0 or r["fidelity_ratio"] <= a.guard)]
        if not recs:
            recs = [r for r in src["arms"] if r["editor"] == ed and np.isfinite(r[key])]
        b = max(recs, key=lambda r: r[key])
        out[ed] = (int(b["point"]), float(b["alpha"]), b["dims"],
                   float(b["fidelity_ratio"]), float(b[key]))
    return out


def blackout_label(i: int) -> str:
    """The blackout that matters for this case, not the min-max of every hidden frame in
    the sequence (several blackouts per sequence is the norm)."""
    if k[i] > 0:
        return f"hidden {EF - k[i]}–{EF - 1}"          # the run that ends at the edit frame
    hid = np.where(~vis_e[i])[0]
    if not hid.size:
        return "never hidden"
    after = hid[hid >= EF]
    return f"visible at EF · next blackout {after.min()}" if after.size else "visible at EF"

# ── the drawing transform: widen the two marker rays ─────────────────────────────────
W = a.edge_width

def widen(arr: np.ndarray) -> np.ndarray:
    """Repeat ray 0 and the last ray W times each, so a 1-px marker is visible. Applied
    identically to every column, so an absent marker stays absent."""
    left = np.repeat(arr[..., :1], W, axis=-1)
    right = np.repeat(arr[..., -1:], W, axis=-1)
    return np.concatenate([left, arr[..., 1:-1], right], axis=-1)

FENCE = "#8a93a6"

for subset in a.subsets:
    sel_all = SUBSETS[subset]
    sel = sel_all[: a.n]                               # index order — no cherry-picking
    if len(sel) == 0:
        print(f"  {subset}: no cases"); continue
    ARMS = arms_for(subset)
    b = dwb.load_bench(model, n=len(sel), target="full", basis_name=a.basis,
                       data_dir=eval_dir, select=sel)
    rolls = {"unsteered": dwa.unsteered_rollout(model, b)}
    (pt, al, dims, fid, sei) = ARMS["PI"]
    rolls[f"PI ({dims}·pt{pt} α{al:g}, fid {fid:.2f})\n192-case EI {sei:+.3f}"] = (
        dwa.pinv_rollout(model, b, lin[pt][0], pt, al, space="zspace", dims=dims))
    (pt, al, dims, fid, sei) = ARMS["GS"]
    rolls[f"GS ({dims}·pt{pt} α{al:g}, fid {fid:.2f})\n192-case EI {sei:+.3f}"] = (
        dwa.grad_steer_rollout(model, b, mlp, pt, al, dims=dims))
    # per-case Edit Index at step 0, and the subset means the table quotes
    per_case = {nm: edit_index_per_case(r[:, 0], b.zones.gt_edited, b.zones.gt_unedited,
                                        b.zones.differing) for nm, r in rolls.items()}
    means = {nm: float(np.nanmean(v)) for nm, v in per_case.items()}

    ctx = b.obs[:, EF - a.n_ctx: EF]
    shift = W - 1                                       # ray r is drawn at x = r + shift

    def cx(mask):
        i = np.where(mask)[0]
        return i.mean() + shift if i.size else np.nan

    fig = waterfall_grid(
        columns={nm: widen(r) for nm, r in rolls.items()},
        context=widen(ctx),
        gt=widen(b.gt_roll),
        title=(f"{a.run} — {subset} cases (n={len(sel)} of {len(sel_all)} available"
               + (f", staleness k ≥ {a.min_k}" if subset == "reappearance" else "")
               + f") · EF={EF}, {K_ROLL}-step rollout, basis={a.basis}, {inst}\n"
               f"edge rays 0 and 127 (the 0.5 blink markers) drawn {W}× wide, inside the "
               f"grey fences — the same transform in every column"
               + ("\nno ghost locator on these rows by construction: the edited object was "
                  "HIDDEN at EF-1, so it vacates no rays" if subset == "reappearance" else "")),
        sample_idx=range(len(sel)),
        target_x=np.array([cx(b.zones.target[i]) for i in range(len(sel))]),
        ghost_x=np.array([cx(b.zones.ghost[i]) for i in range(len(sel))]),
        metrics=means,
        metric_label="drawn rows: Edit Index",
    )
    # fence off the widened marker track in every cell, and label the drawn rows
    width = b.gt_roll.shape[-1] + 2 * (W - 1)
    for ax in fig.axes:
        ax.axvline(W - 0.5, color=FENCE, lw=0.8, alpha=0.85)
        ax.axvline(width - W - 0.5, color=FENCE, lw=0.8, alpha=0.85)
    for r, i in enumerate(sel):
        fig.axes[r * (len(rolls) + 1)].set_ylabel(
            f"case {i} · k={k[i]}\n{blackout_label(i)}\ntime ↓", color="#c9d1e0", fontsize=7.5)
    path = out_dir / f"waterfall_{subset}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n{subset}: {len(sel)} of {len(sel_all)} cases drawn → {path.relative_to(REPO)}")
    print(f"  arms (fidelity guard {a.guard or 'off'}): "
          + "  ".join(f"{ed} pt{v[0]} α{v[1]:g} {v[2]} fid {v[3]:.2f}" for ed, v in ARMS.items()))
    print("  Edit Index over the DRAWN rows: "
          + "  ".join(f"{nm.split(' ')[0]} {m:+.3f}" for nm, m in means.items()))
    print("  the same arms over the FULL 192-case subset: "
          + f"unsteered {SUB[subset]['unedited']['edit_index']:+.3f}  "
          + "  ".join(f"{ed} {v[4]:+.3f}" for ed, v in ARMS.items())
          + "   ← the result; the drawn rows are 6 examples, not evidence")
    hdr = f"  {'case':>6} {'k':>3} {'blackout':>12} " + " ".join(f"{nm.split(' ')[0]:>10}" for nm in rolls)
    print(hdr)
    for r, i in enumerate(sel):
        span = blackout_label(i).replace("hidden ", "")
        print(f"  {i:>6} {k[i]:>3} {span:>12} "
              + " ".join(f"{per_case[nm][r]:>+10.3f}" for nm in rolls))

    # what the markers do in the drawn window: GT vs each column, on the two edge rays
    mk = []
    for r, i in enumerate(sel):
        pre, post = bk.transitions(vis[i][None])
        frames = [(t, j) for t in range(EF - a.n_ctx, EF + K_ROLL) for j in range(N_OBJ)
                  if (pre[0, t, j] or post[0, t, j])]
        mk.append((i, frames))
    print("  markers in the drawn window (frame, object):")
    for i, frames in mk:
        print(f"    case {i}: {frames if frames else '—'}")
