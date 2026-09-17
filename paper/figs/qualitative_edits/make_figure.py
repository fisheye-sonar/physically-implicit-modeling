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
each run's cache, the writes from ``arms`` (first step of the scored rollout). Output:
``qualitative_edits_seed<seed>.{pdf,png}`` beside this script.

    python paper/figs/qualitative_edits/make_figure.py --seed 7
    python paper/figs/qualitative_edits/make_figure.py --seed 7 --find   # advance the seed until the
                                                         # teleport changes a categorical tile everywhere
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
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
from pim.models import load_checkpoint  # noqa: E402

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


def bench_for(model, sc: dict, obs, clean, vis, sim: dict, target: str, basis: str):
    n1 = lambda x: np.asarray(x)[None]  # noqa: E731
    a = dwb.bench_from_arrays(n1(obs), n1(sc["pos"]).astype(np.float32), n1(sc["vel"]).astype(np.float32),
                              np.array([sc["edit_object"]]), n1(clean), sim,
                              None if vis is None else n1(vis), target=target, basis_name=basis)
    return dwb.bench_of(model, a), a


def best_arm(scores: dict, block: str, editor: str) -> dict:
    return scores["bases"][block]["best"][editor]


@torch.no_grad()
def predictions(model, run_dir: Path, inst: str, sc: dict, obs, clean, vis, sim: dict, block: tuple) -> dict:
    """Next-step frames for unedited / PI / GS / IM at the run's scored best arms on one block."""
    target, basis = block
    scores = json.loads((run_dir / "scores.json").read_text())
    b, a = bench_for(model, sc, obs, clean, vis, sim, target, basis)
    recipe = dwa.probe_recipe(target, inst, n_seq=30_000)
    cache = run_dir / "probes"
    out = {"unedited": dwa.unsteered_rollout(model, b)[0, 0], "changes_tile": bool(a["change_mask"].any())}
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
    _, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, basis)
    # the inverse map is fitted on the FULL state under the full/30k recipe in every block (the
    # scorer's inverse_discworld); only its basis follows the block — the cache key depends on it
    gen = dwa.iter_inverse_maps(model, basis_name=basis, points=[im["point"]], cache_dir=cache, log=None,
                                **dwa.probe_recipe("full", inst, n_seq=30_000))
    _, g, _, _ = next(gen)
    gen.close()                                   # frees the retrieval bank on the GPU
    dwa.as_activations(model, im["point"])
    out["IM"] = model.decode_with_edit(b.state, im["point"], inverse_overwrite(g, torch.from_numpy(s_post).to(DEV)))[0].cpu().numpy()
    out["arms"] = {ed: (arm[ed]["point"], arm[ed]["alpha"]) for ed in EDITORS}
    return out


def build(seed: int, context: int) -> dict:
    cfgs = {inst: sim_config(inst) for _, inst, _ in VARIANTS}
    base = max(cfgs.values(), key=lambda c: c.radius)      # the tightest geometry hosts the scenario
    sc = scenario(seed, base)
    cols = {}
    for name, inst, run in VARIANTS:
        run_dir = REPO / "runs" / run
        model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
        model.eval()
        obs, clean, vis = render_under(sc, cfgs[inst], seed)
        sim = dataclasses.asdict(dataclasses.replace(cfgs[inst], seed=int(seed), n_objects=N_OBJ))
        cols[name] = {"context": obs[EF - context:EF], "gt": clean[EF],
                      "cont": predictions(model, run_dir, inst, sc, obs, clean, vis, sim, CONT),
                      "cat": predictions(model, run_dir, inst, sc, obs, clean, vis, sim, CAT)}
        print(f"  {name:<9} rays {obs.shape[1]:>3}  fac tile changes: {cols[name]['cat']['changes_tile']}  "
              f"arms cont {cols[name]['cont']['arms']}  cat {cols[name]['cat']['arms']}", flush=True)
        del model
        torch.cuda.empty_cache()
    return {"seed": seed, "edit_object": sc["edit_object"], "cols": cols}


# ── drawing ──────────────────────────────────────────────────────────────────────────
STRIP = 1.0          # height of one single-frame row, in waterfall-frame units
GAP, BIGGAP = 0.35, 1.6
CMAP = "gray_r"      # 0 = white (empty ray), dark = a disc


def _strip(ax, row: np.ndarray, vmin=0.0, vmax=1.0):
    ax.imshow(np.asarray(row)[None, :], cmap=CMAP, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5); sp.set_edgecolor("#9a9a9a")


def draw(fig_data: dict, context: int, out: Path, title_size: float = 16, label_size: float = 11):
    names = [v[0] for v in VARIANTS]
    rows = ([("waterfall", context)] + [("gap", GAP), ("Unedited", STRIP), ("gap", GAP), ("Ground truth", STRIP), ("gap", BIGGAP)]
            + [x for ed in EDITORS for x in (("cont:" + ed, STRIP), ("gap", GAP))][:-1] + [("gap", BIGGAP)]
            + [x for ed in EDITORS for x in (("cat:" + ed, STRIP), ("gap", GAP))][:-1])
    heights = [h for _, h in rows]
    ncol = len(names)
    fig = plt.figure(figsize=(2.55 * ncol + 1.9, 0.42 * sum(heights) + 1.0), facecolor="white")
    gs = GridSpec(len(rows), ncol, figure=fig, height_ratios=heights, left=0.17, right=0.995,
                  top=0.94, bottom=0.01, wspace=0.10, hspace=0.0)
    first = {}
    for c, name in enumerate(names):
        col = fig_data["cols"][name]
        for r, (kind, _) in enumerate(rows):
            if kind == "gap":
                continue
            ax = fig.add_subplot(gs[r, c])
            if kind == "waterfall":
                ax.imshow(col["context"], cmap=CMAP, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.5); sp.set_edgecolor("#9a9a9a")
                ax.set_title(name, fontsize=title_size, pad=8)
            elif kind == "Unedited":
                _strip(ax, col["cont"]["unedited"])
            elif kind == "Ground truth":
                _strip(ax, col["gt"])
            else:
                blk, ed = kind.split(":")
                _strip(ax, col[blk][ed])
            if c == 0:
                first[kind] = ax
    # row labels on the first column, group labels further left
    for kind, ax in first.items():
        if kind == "waterfall":
            lab = f"last {context} frames"
        elif ":" in kind:
            lab = kind.split(":")[1]
        else:
            lab = kind
        ax.text(-0.04, 0.5, lab, transform=ax.transAxes, ha="right", va="center", fontsize=label_size)
    for blk, text in (("cont", "Continuous\npositions"), ("cat", "Categorical\npositions")):
        axes = [first[f"{blk}:{ed}"] for ed in EDITORS]
        y0 = axes[-1].get_position().y0; y1 = axes[0].get_position().y1
        fig.text(0.035, (y0 + y1) / 2, text, ha="center", va="center", rotation=90, fontsize=label_size + 1)
        fig.add_artist(plt.Line2D([0.065, 0.065], [y0, y1], transform=fig.transFigure, color="#444", lw=0.9))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=8, help="frames of context above the edit")
    ap.add_argument("--find", action="store_true",
                    help="advance the seed until the teleport changes a categorical tile on every variant")
    ap.add_argument("--max-tries", type=int, default=50)
    a = ap.parse_args()
    seed = a.seed
    for attempt in range(a.max_tries if a.find else 1):
        print(f"seed {seed}", flush=True)
        data = build(seed, a.context)
        ok = all(c["cat"]["changes_tile"] for c in data["cols"].values())
        if ok or not a.find:
            break
        seed += 1
    else:
        raise SystemExit("no seed in range changes a categorical tile on every variant")
    if not ok:
        print("⚠ on some variant the teleport does not change a factorised tile — the categorical rows there "
              "ask for no change (use --find)")
    out = HERE / f"qualitative_edits_seed{seed}"
    draw(data, a.context, out)
    json.dump({"seed": seed, "edit_object": data["edit_object"],
               "arms": {n: {"cont": c["cont"]["arms"], "cat": c["cat"]["arms"]} for n, c in data["cols"].items()},
               "changes_tile": {n: c["cat"]["changes_tile"] for n, c in data["cols"].items()}},
              open(out.with_suffix(".json"), "w"), indent=1)
    print("→", out.with_suffix(".pdf").relative_to(REPO), "and .png / .json")
