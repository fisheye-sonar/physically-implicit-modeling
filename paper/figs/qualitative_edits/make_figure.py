"""Qualitative edits across the ray-world variants — one matched scenario, next-step predictions.

One trajectory (two discs, a teleport of one of them at the edit frame) is generated from
``--seed`` under the most restrictive geometry (disc radius 1.0, the coarse-ray family's) and
rendered under EVERY listed variant's own renderer, so the columns show the same world seen
through 128 / 16 / 8 / 5 rays (and with blackouts on the blink variant). Each column: the last
``--context`` observed frames above the edit (a waterfall, time downward), then single-frame
NEXT-STEP predictions — unedited, the clean ground truth, and each editor's write at the run's
scored best arm on the continuous (full-state, Cartesian) and the categorical (factorised
appearance, frustum) targets. The write targets the PRE-dynamics state, as in the scorer.

Everything canonical comes from ``pim``: the scenario from the edit-set generator, the bench
from ``bench.bench_from_arrays`` (the scorer's own construction), probes / inverse maps from
each run's cache (never fitted here), the writes from ``arms`` (first step of the scored rollout).
On a categorical block the IM write goes through the block's OWN inverse map (one-hot labels +
Cartesian velocity, 2026-09-20); a block with no IM arm (Standard, Blink) draws a blank cell.
Output: ``qualitative_edits_seed<seed>.{pdf,png}`` beside this script.

The scenario filter (2026-09-21): a seed is drawn only if its teleport changes at least one ray of the clean
5-ray frame at the edit frame (``visible_change``: the scorer's differing-ray zone under the dw-5ray renderer
is non-empty), so every column of every figure shows an edit that is visible even on the coarsest sensor.

    python paper/figs/qualitative_edits/make_figure.py --seed 7
    python paper/figs/qualitative_edits/make_figure.py --seed 7 --find   # advance the seed until it passes the filter
    python paper/figs/qualitative_edits/make_figure.py --passing 6       # the first six passing seeds: the first beside
                                                         # this script, the rest under more_seeds/seed<k>/
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # paper/figs
import paper_style as ps  # noqa: E402

ps.apply()                                  # one look for every paper figure (Arial, TrueType, white page)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.inverse import inverse_overwrite  # noqa: E402
from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, N_OBJ, full_state_pair  # noqa: E402
from pim.environments.discworld.blink import blink_schedule  # noqa: E402
from pim.environments.discworld.config import SimConfig  # noqa: E402
from pim.environments.discworld.edits_dataset import _generate_one_edit  # noqa: E402
from pim.environments.discworld.renderer import render_scene  # noqa: E402
from pim.environments.discworld.sim import Scene  # noqa: E402
from pim.metrics.selection import best_arm as _select_arm  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.inverse import encode_categorical_state  # noqa: E402

HERE = Path(__file__).resolve().parent
# ── the columns: (display name, instance, run) — edit this list to change the figure ──
VARIANTS = [
    ("Standard", "dw-noiseless", "noise_ablation/L-dw-noiseless-20m"),
    ("Blink", "dw-blink", "blink_ablation/L-dw-blink-20m"),
    ("16-ray", "dw-16ray", "ray_ablation/L-dw-16ray-20m"),
    ("8-ray", "dw-8ray", "ray_ablation/L-dw-8ray-20m"),
    ("5-ray", "dw-5ray", "ray_ablation/L-dw-5ray-20m"),
]
CONT = ("full", "cartesian")            # continuous positions: the full-state regression block
CAT = ("appearance-fac", "frustum")     # categorical positions: the factorised appearance block
EDITORS = ("PI", "GS", "IM")
DEV = dwb.DEV


def sim_config(inst: str) -> SimConfig:
    """The instance's own sim config, from its edit split's contract."""
    with h5py.File(layout.edits_file("discworld", inst), "r") as f:
        return SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])


def scenario(seed: int, base: SimConfig) -> dict:
    """One teleport case from the shared generator under ``base`` (radius 1.0 — valid under
    every variant, the smaller discs included). Returns the post-edit trajectory."""
    r = _generate_one_edit((seed, base, N_OBJ, EF, True, 2000))
    return {"pos": r["positions"][:, :N_OBJ].astype(np.float64), "vel": r["velocities"][:, :N_OBJ].astype(np.float64),
            "refl": r["reflectivities"][:N_OBJ].astype(np.float64), "colors": r["colors"][:N_OBJ].astype(np.float64),
            "edit_object": int(r["edit_object"])}


def render_under(sc: dict, cfg: SimConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(obs, clean, visible) for the scenario under one variant's geometry. Blink variants get
    the schedule the seed implies, forced visible from EF−1 on so the edit is observable."""
    cfg = dataclasses.replace(cfg, seed=int(seed), n_objects=N_OBJ)
    scene = Scene(positions=sc["pos"], velocities=sc["vel"], radii=np.full(N_OBJ, cfg.radius),
                  colors=sc["colors"], reflectivities=sc["refl"], config=cfg)
    vis = blink_schedule(cfg, N_OBJ)
    if vis is not None:
        vis = vis.copy(); vis[EF - 1:] = True
    _, _, obs = render_scene(scene, visible=vis)
    _, _, clean = render_scene(dataclasses.replace(scene, config=dataclasses.replace(cfg, obs_noise_std=0.0)), visible=vis)
    return obs.astype(np.float32), clean.astype(np.float32), vis


def bench_arrays_for(sc: dict, obs, clean, vis, sim: dict, target: str, basis: str) -> dict:
    """The scorer's bench construction (``bench_from_arrays``: targets, change mask, ray zones) for the one
    scenario; no model involved."""
    n1 = lambda x: np.asarray(x)[None]  # noqa: E731
    return dwb.bench_from_arrays(n1(obs), n1(sc["pos"]).astype(np.float32), n1(sc["vel"]).astype(np.float32),
                                 np.array([sc["edit_object"]]), n1(clean), sim,
                                 None if vis is None else n1(vis), target=target, basis_name=basis)


def bench_for(model, sc: dict, obs, clean, vis, sim: dict, target: str, basis: str):
    a = bench_arrays_for(sc, obs, clean, vis, sim, target, basis)
    return dwb.bench_of(model, a), a


def sim_dict(cfg: SimConfig, seed: int) -> dict:
    return dataclasses.asdict(dataclasses.replace(cfg, seed=int(seed), n_objects=N_OBJ))


def base_config(cfgs: dict | None = None) -> SimConfig:
    """The geometry every scenario is generated under: the tightest among ``VARIANTS`` (radius 1.0, the
    coarse-ray family's), so the world is valid under every variant."""
    cfgs = cfgs or {inst: sim_config(inst) for _, inst, _ in VARIANTS}
    return max((cfgs[inst] for _, inst, _ in VARIANTS), key=lambda c: c.radius)


# ── the scenario filter (Sevan, 2026-09-21: "only show examples which change for all of them") ──
FILTER_INST = "dw-5ray"      # the coarsest renderer: a scenario is drawn only if its teleport changes THIS frame


def differing_rays(sc: dict, cfg: SimConfig, seed: int) -> np.ndarray:
    """The rays on which the clean post-edit frame at EF differs from the clean unedited frame, under
    ``cfg``'s renderer: the scorer's own ``differing`` zone (``build_edit_zones`` inside
    ``bench_from_arrays``), the support the Edit Index is scored over. Empty: the teleport is invisible."""
    obs, clean, vis = render_under(sc, cfg, seed)
    a = bench_arrays_for(sc, obs, clean, vis, sim_dict(cfg, seed), *CONT)
    return np.flatnonzero(a["zones"].differing[0])


def visible_change(seed: int, inst: str = FILTER_INST) -> np.ndarray:
    """``differing_rays`` of the seed's scenario under ``inst``'s renderer (CPU, no model)."""
    return differing_rays(scenario(seed, base_config()), sim_config(inst), seed)


def passing_seeds(n: int, start: int = 0, inst: str = FILTER_INST, max_tries: int = 500) -> list[int]:
    """The first ``n`` seeds from ``start`` whose teleport changes at least one ray of the clean frame under
    ``inst`` (default: the 5-ray renderer). The selection rule of every drawn scenario since 2026-09-21."""
    out = [s for s in range(start, start + max_tries) if len(visible_change(s, inst))][:n]
    if len(out) < n:
        raise SystemExit(f"only {len(out)} of {n} seeds in [{start}, {start + max_tries}) change the {inst} frame")
    return out


def best_arm(scores: dict, block: str, editor: str) -> dict | None:
    """The arm the TABLES report for this editor (``pim.metrics.selection.best_arm``: the best Edit Index
    inside the fidelity guard, the unguarded best only if there is none) — so the figure draws the write
    whose numbers the paper quotes. Until 2026-09-19 this read the scorer's unguarded ``best``. ``None``
    where the block carries no arm for the editor at all (since 2026-09-20 the categorical IM on every
    instance outside the ray family): the table cell is blank, and so is the figure's."""
    B = scores["bases"][block]
    return _select_arm(B["arms"], editor, "edit_index") or B["best"].get(editor)


@contextlib.contextmanager
def _cache_hits_only():
    """The categorical inverse map is a ~30 min streamed fit over 200k sequences; this figure may only USE
    the map the scorer cached in the run's ``probes/``. A cache miss fails loudly instead of fitting."""
    def refuse(*a, **k):
        raise RuntimeError("categorical inverse map not in the run's probe cache — this figure never fits one")
    saved, dwa.collect_residuals = dwa.collect_residuals, refuse
    try:
        yield
    finally:
        dwa.collect_residuals = saved


@torch.no_grad()
def predictions(model, run_dir: Path, inst: str, sc: dict, obs, clean, vis, sim: dict, block: tuple) -> dict:
    """Next-step frames for unedited / PI / GS / IM at the run's scored best arms on one block. An editor
    with no arm in the block (``best_arm`` None) gets ``None`` — drawn as a blank cell."""
    target, basis = block
    scores = json.loads((run_dir / "scores.json").read_text())
    b, a = bench_for(model, sc, obs, clean, vis, sim, target, basis)
    recipe = dwa.probe_recipe(target, inst, n_seq=30_000)
    cache = run_dir / "probes"
    cx = lambda m: float(np.where(np.asarray(m))[0].mean()) if np.asarray(m).any() else float("nan")  # noqa: E731
    out = {"unedited": dwa.unsteered_rollout(model, b)[0, 0], "changes_tile": bool(a["change_mask"].any()),
           "ghost_x": cx(a["zones"].ghost[0]), "target_x": cx(a["zones"].target[0]),   # ray centres at the edit frame
           "differing_rays": np.flatnonzero(a["zones"].differing[0])}                  # the Edit Index support
    arm = {ed: best_arm(scores, target if target != "full" else basis, ed) for ed in EDITORS}
    lin = dwa.fit_probes(model, target=target, family="linear", basis_name=basis, cache_dir=cache,
                         log=None, require_cached=True, **recipe)
    mlp = dwa.fit_probes(model, target=target, family="mlp", basis_name=basis, cache_dir=cache,
                         log=None, require_cached=True, **recipe)
    pi = arm["PI"]
    out["PI"] = dwa.pinv_rollout(model, b, lin[pi["point"]][0], pi["point"], pi["alpha"],
                                 space="zspace", dims=pi.get("dims", "all"))[0, 0]
    gs = arm["GS"]
    out["GS"] = dwa.grad_steer_rollout(model, b, mlp, gs["point"], gs["alpha"], dims=gs.get("dims", "all"))[0, 0]
    im = arm["IM"]
    if im is None:
        out["IM"] = None                          # no IM arm in this block: the cell is blank, like the table's
    else:
        # The write ``arms.inverse_arms`` makes at the arm's point, h' = g(s_post), with the map the block's own
        # probes read (2026-09-20). REGRESSION block: g is the continuous full-state map of the block's basis
        # (the scorer's full/30k recipe). CATEGORICAL block: g is the categorical map — the target's own labels
        # one-hot (``Bench.tgt``) plus the discs' Cartesian velocity, the forward probe's recipe
        # (``probe_recipe(target)`` = ``GRID_PROBE_RECIPE``) — exactly as ``inverse_discworld`` calls it.
        cat = target != "full"
        _, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, "cartesian" if cat else basis)
        s_post = torch.from_numpy(s_post).to(DEV)
        if cat:
            with _cache_hits_only():
                gen = dwa.iter_inverse_maps(model, basis_name=basis, target=target, points=[im["point"]],
                                            cache_dir=cache, log=None, **recipe)
                _, g, _, st = next(gen)
            gen.close()
            s_post = encode_categorical_state(b.tgt, s_post[:, 2 * N_OBJ:], st["n_classes"])
        else:
            gen = dwa.iter_inverse_maps(model, basis_name=basis, points=[im["point"]], cache_dir=cache, log=None,
                                        **dwa.probe_recipe("full", inst, n_seq=30_000))
            _, g, _, _ = next(gen)
            gen.close()                           # frees the retrieval bank on the GPU
        dwa.as_activations(model, im["point"])
        out["IM"] = model.decode_with_edit(b.state, im["point"], inverse_overwrite(g, s_post))[0].cpu().numpy()
    out["arms"] = {ed: None if arm[ed] is None else (arm[ed]["point"], arm[ed]["alpha"]) for ed in EDITORS}
    return out


def build(seed: int, context: int, variants=None) -> dict:
    """Predictions for every (name, instance, run) in ``variants`` (default ``VARIANTS``). The scenario is
    always generated under the appendix's base geometry (the tightest among ``VARIANTS``), so an extra
    variant — the main text's 128-ray Standard — sees the same world as the appendix columns of that seed."""
    variants = list(VARIANTS if variants is None else variants)
    cfgs = {inst: sim_config(inst) for _, inst, _ in VARIANTS + variants}
    sc = scenario(seed, base_config(cfgs))
    cols = {}
    for name, inst, run in variants:
        run_dir = REPO / "runs" / run
        model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
        model.eval()
        obs, clean, vis = render_under(sc, cfgs[inst], seed)
        sim = sim_dict(cfgs[inst], seed)
        cols[name] = {"context": obs[EF - context:EF], "gt": clean[EF],
                      "cont": predictions(model, run_dir, inst, sc, obs, clean, vis, sim, CONT),
                      "cat": predictions(model, run_dir, inst, sc, obs, clean, vis, sim, CAT)}
        print(f"  {name:<9} rays {obs.shape[1]:>3}  changed rays {len(cols[name]['cont']['differing_rays']):>3}  "
              f"fac tile changes: {cols[name]['cat']['changes_tile']}  "
              f"arms cont {cols[name]['cont']['arms']}  cat {cols[name]['cat']['arms']}", flush=True)
        del model
        torch.cuda.empty_cache()
    return {"seed": seed, "edit_object": sc["edit_object"], "cols": cols}


# ── drawing ──────────────────────────────────────────────────────────────────────────
# Observations are drawn exactly as the canonical waterfall draws them (pim/figures/waterfall.py):
# ``gray`` on the dark panel background, fixed 0–1 range, nearest interpolation. The page stays
# white and the text black. The ``diff`` variant draws the six edit rows as prediction − ground
# truth on the canonical signed-error map (red = under-prediction, green = over, zero = background).
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402

from pim.figures.waterfall import DARK_BG, DIFF_CMAP, GHOST_C, TARGET_C  # noqa: E402

STRIP = 1.0          # height of one single-frame row, in waterfall-frame units
CTX_ROW = 0.55       # height of one waterfall (context) frame, in the same units — squashed against the strips
COL_W = 3.3          # inches per column
PAIR_DIFF = 0.75     # the error strip under a prediction in the "paired" variant
GAP, BIGGAP = 0.22, 0.8
TEXT = ps.TEXT
FRAME = ps.FRAME     # thin panel border


OVERLAY_ALPHA = 0.85     # a fully wrong ray keeps a trace of its own grey under the tint
OVERLAY_CMAP = LinearSegmentedColormap.from_list("pim_overlay", [GHOST_C, "#8c8c8c", TARGET_C])


def error(pred: np.ndarray, gt: np.ndarray, raw: bool = False) -> np.ndarray:
    """prediction − truth. By default the prediction is CLIPPED to the observation range [0, 1] first,
    so the tint reflects what the grey panel shows: a raw output of −0.85 on an empty ray is drawn
    black, as the truth is, and is not an error to the eye. The scorer does not clip (``zone_rmse``
    uses the raw rollout, so that −0.85 IS error in the Edit Index); ``raw=True`` reproduces that."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    return (pred if raw else np.clip(pred, 0.0, 1.0)) - gt


def overlay_rgb(pred: np.ndarray, gt: np.ndarray, scale: float, signed: bool = True,
                gamma: float = 1.0, raw: bool = False) -> np.ndarray:
    """The prediction drawn in grey, each ray tinted in proportion to (|prediction − truth| / scale)^gamma
    — a perfect ray is just its grey. ``signed``: red for under-prediction, green for over;
    otherwise red alone, by absolute error. ``gamma`` > 1 mutes small errors (an intensity error of
    0.15 is invisible in grey but tints at gamma 1), < 1 amplifies them. ``raw``: see ``error``."""
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    g = np.clip(pred, 0.0, 1.0)[..., None].repeat(3, -1)
    err = error(pred, gt, raw)
    a = OVERLAY_ALPHA * np.clip(np.abs(err) / scale, 0.0, 1.0)[..., None] ** gamma
    red = np.array(to_rgb(GHOST_C))
    tint = np.where((err < 0)[..., None], red, np.array(to_rgb(TARGET_C))) if signed else red
    return (1.0 - a) * g + a * tint


def _panel(ax, img: np.ndarray, *, diff: bool = False, diff_scale: float = 1.0, rgb: bool = False):
    img = np.asarray(img)
    if img.ndim == (2 if rgb else 1):
        img = img[None]
    if rgb:
        ax.imshow(img, aspect="auto", interpolation="nearest")
    elif diff:
        ax.imshow(img, cmap=DIFF_CMAP, vmin=-diff_scale, vmax=diff_scale, aspect="auto", interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    _frame(ax)


def _blank(ax):
    """An empty cell — the block carries no arm for this editor (the table's blank): the spot is left fully empty
    (no panel, no frame; Sevan, round 3), the axes only holds the place so the row label and gaps stay put."""
    ax.set_axis_off()


def _frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor(FRAME)


ORIGIN_C, DEST_C = ps.ORIGIN_C, ps.DEST_C   # cyan = where the edited disc came from, pink = where it was moved to


def draw(fig_data: dict, context: int, out: Path, *, mode: str = "obs", diff_scale: float = 1.0,
         tint_gamma: float = 1.0, locators: bool = True, raw_error: bool = False,
         title_size: float = 24, label_size: float = 17):
    names = [v[0] for v in VARIANTS]
    # "paired": every edit row is its plain prediction with the signed-error strip directly beneath (no gap)
    def edit_rows(blk):
        out = []
        for ed in EDITORS:
            out.append((f"{blk}:{ed}", STRIP))
            if mode == "paired":
                out.append((f"{blk}:{ed}:diff", PAIR_DIFF))
            out.append(("gap", GAP))
        return out[:-1]
    rows = ([("waterfall", context * CTX_ROW)] + [("gap", GAP), ("Unedited", STRIP), ("gap", GAP), ("Ground truth", STRIP), ("gap", BIGGAP)]
            + edit_rows("cont") + [("gap", BIGGAP)] + edit_rows("cat"))
    heights = [h for _, h in rows]
    ncol = len(names)
    fig = plt.figure(figsize=(COL_W * ncol + 1.6, 0.40 * sum(heights) + 1.2), facecolor="white")
    bar = mode in ("diff", "overlay", "paired")   # "abs" (red by |error|) carries no bar — the caption explains it
    gs = GridSpec(len(rows), ncol, figure=fig, height_ratios=heights, left=0.15, right=0.945 if bar else 0.995,
                  top=0.94, bottom=0.01, wspace=0.10, hspace=0.0)
    first = {}
    for c, name in enumerate(names):
        col = fig_data["cols"][name]
        for r, (kind, _) in enumerate(rows):
            if kind == "gap":
                continue
            ax = fig.add_subplot(gs[r, c])
            blank = False
            if kind == "waterfall":
                _panel(ax, col["context"])
                ax.set_title(name, fontsize=title_size, pad=8, color=TEXT)
            elif kind == "Unedited":
                _panel(ax, col["cont"]["unedited"])
            elif kind == "Ground truth":
                _panel(ax, col["gt"])                             # the reference every row below is judged against
            else:
                blk, ed, *sub = kind.split(":")
                blank = col[blk][ed] is None                      # no arm in this block (the table's blank cell)
                if blank:
                    _blank(ax)
                elif sub:                                         # the paired variant's error strip
                    _panel(ax, error(col[blk][ed], col["gt"], raw_error), diff=True, diff_scale=diff_scale)
                elif mode == "diff":
                    _panel(ax, error(col[blk][ed], col["gt"], raw_error), diff=True, diff_scale=diff_scale)
                elif mode in ("overlay", "abs"):
                    _panel(ax, overlay_rgb(col[blk][ed], col["gt"], diff_scale, signed=(mode == "overlay"),
                                           gamma=tint_gamma, raw=raw_error), rgb=True)
                else:
                    _panel(ax, col[blk][ed])
            if locators and kind != "waterfall" and not blank:
                for key, colr in (("ghost_x", ORIGIN_C), ("target_x", DEST_C)):
                    x = col["cont"].get(key, float("nan"))
                    if np.isfinite(x):
                        ax.axvline(x, color=colr, lw=1.3, alpha=0.95)
            if c == 0:
                first[kind] = ax
    # row labels just left of the first column; the group labels against them, not the page edge
    x_lab = first["waterfall"].get_position().x0 - 0.006
    for kind, ax in first.items():
        if kind.endswith(":diff"):
            continue
        if kind == "waterfall":
            lab = f"last {context} frames"
        elif ":" in kind:
            lab = kind.split(":")[1]
        else:
            lab = {"Unedited": "Unedited Pred"}.get(kind, kind)
        lo = first.get(kind + ":diff", ax)                          # paired: centre on both strips
        y = (lo.get_position().y0 + ax.get_position().y1) / 2
        fig.text(x_lab, y, lab, ha="right", va="center", fontsize=label_size, color=TEXT,
                 fontweight="bold" if kind == "Ground truth" else "normal")
    x_line = x_lab - 0.040
    for blk, text in (("cont", "Continuous\npositions"), ("cat", "Categorical\npositions")):
        axes = [first[f"{blk}:{ed}"] for ed in EDITORS]
        y0 = first.get(f"{blk}:{EDITORS[-1]}:diff", axes[-1]).get_position().y0; y1 = axes[0].get_position().y1
        fig.add_artist(plt.Line2D([x_line, x_line], [y0, y1], transform=fig.transFigure, color=TEXT, lw=0.9))
        fig.text(x_line - 0.030, (y0 + y1) / 2, text, ha="center", va="center", rotation=90,
                 fontsize=label_size + 1, color=TEXT)
    if bar:
        import matplotlib as mpl
        y0 = first.get(f"cat:{EDITORS[-1]}:diff", first["cat:" + EDITORS[-1]]).get_position().y0
        y1 = first["cont:" + EDITORS[0]].get_position().y1
        cax = fig.add_axes([0.958, y0, 0.010, y1 - y0])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=OVERLAY_CMAP if mode == "overlay" else DIFF_CMAP, orientation="vertical",
                                       norm=mpl.colors.Normalize(-diff_scale, diff_scale))
        cb.set_ticks([-diff_scale, 0, diff_scale]); cb.ax.tick_params(labelsize=12, colors=TEXT, length=2)
        cb.outline.set_edgecolor(FRAME); cb.outline.set_linewidth(0.5)
        cb.set_label("red = under-prediction, green = over", fontsize=14, color=TEXT, rotation=90, labelpad=6)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def cached(seed: int, context: int, redraw: bool) -> dict:
    """One seed's predictions: the ``.scratch/`` cache when ``redraw`` and it exists, else built (GPU, one model at
    a time) and cached. ``_catim`` (2026-09-21): the categorical blocks' IM through the categorical map."""
    cache = REPO / ".scratch" / f"qualitative_edits_catim_seed{seed}_ctx{context}.pkl"
    if redraw and cache.exists():
        return pickle.load(open(cache, "rb"))
    data = build(seed, context)
    cache.parent.mkdir(exist_ok=True)
    pickle.dump(data, open(cache, "wb"))
    return data


def render(data: dict, context: int, out_dir: Path, **kw) -> Path:
    """All five modes of one seed's figure and its sidecar, into ``out_dir``."""
    seed = data["seed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"qualitative_edits_seed{seed}"
    draw(data, context, out, **kw)
    for mode in ("diff", "overlay", "abs", "paired"):
        draw(data, context, out.with_name(f"{out.name}_{mode}"), mode=mode, **kw)
    if not all(c["cat"]["changes_tile"] for c in data["cols"].values()):
        print("⚠ on some variant the teleport does not change a factorised tile: the categorical rows there ask for no change")
    json.dump({"seed": seed, "edit_object": data["edit_object"],
               "filter": {"instance": FILTER_INST, "changed_rays": visible_change(seed).tolist(),
                          "rule": "drawn only if the teleport changes at least one ray of the clean 5-ray frame at the edit frame"},
               "differing_rays": {n: c["cont"]["differing_rays"].tolist() for n, c in data["cols"].items()},
               "arms": {n: {"cont": c["cont"]["arms"], "cat": c["cat"]["arms"]} for n, c in data["cols"].items()},
               "changes_tile": {n: c["cat"]["changes_tile"] for n, c in data["cols"].items()}},
              open(out.with_suffix(".json"), "w"), indent=1)
    print("→", out.with_suffix(".pdf"), "(+ _diff, _overlay, _abs, _paired) and .png / .json", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=8, help="frames of context above the edit")
    ap.add_argument("--find", action="store_true",
                    help="advance the seed until it passes the scenario filter (the teleport changes the 5-ray frame)")
    ap.add_argument("--passing", type=int, default=0, metavar="N",
                    help="draw the first N seeds that pass the filter: the first beside this script, the rest under more_seeds/seed<k>/")
    ap.add_argument("--out-dir", default=None, help="write the outputs here instead of beside this script")
    ap.add_argument("--redraw", action="store_true", help="reuse the cached predictions (.scratch/) where they exist")
    ap.add_argument("--diff-scale", type=float, default=1.0, help="± range of the error map in the _diff variant")
    ap.add_argument("--tint-gamma", type=float, default=1.0, help="exponent on |error|/scale for the overlay tints")
    ap.add_argument("--no-locators", action="store_true", help="drop the cyan (origin) / pink (destination) lines")
    ap.add_argument("--raw-error", action="store_true",
                    help="tint by the RAW prediction − truth (the scorer's quantity; predictions below 0 or above 1 "
                         "count) instead of the clipped, visible one")
    a = ap.parse_args()
    kw = dict(diff_scale=a.diff_scale, tint_gamma=a.tint_gamma, locators=not a.no_locators, raw_error=a.raw_error)
    if a.passing:                                  # the appendix set: slot k = the k-th passing seed
        seeds = passing_seeds(a.passing)
        dirs = [HERE] + [HERE / "more_seeds" / f"seed{s}" for s in seeds[1:]]
    else:
        seeds = passing_seeds(1, start=a.seed) if a.find else [a.seed]
        dirs = [Path(a.out_dir) if a.out_dir else HERE]
    for seed, out_dir in zip(seeds, dirs):
        print(f"seed {seed}  changed 5-ray rays {visible_change(seed).tolist()}", flush=True)
        render(cached(seed, a.context, a.redraw), a.context, out_dir, **kw)
