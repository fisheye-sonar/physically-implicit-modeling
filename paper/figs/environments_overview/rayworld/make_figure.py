"""Rayworld and its variants: the environment figure pieces (paper ``fig:rayworld_and_variants``).

Real held-out sequences (``datasets/discworld/<inst>/eval/test.h5``) drawn with the canonical
geometry: ray directions exactly as ``renderer.render_frame`` spreads them (uniform in the
tangent of the viewing angle, wall rays dropped when the instance says so), hits and hit depths
from the stored ``obs_id`` / ``obs_depth``, appearance cells from
``grid_target.categorical_target("appearance")``, the matched N-ray strips from
``renderer.render_scene`` under each sibling instance's own ``SimConfig``. No metric, no
re-implemented rendering. CPU only. Outputs land beside this script.

    .pim/bin/python paper/figs/environments_overview/rayworld/make_figure.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))          # paper/figs  ->  paper_style
sys.path.insert(0, str(REPO))
import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from pim.environments import layout  # noqa: E402
from pim.environments.discworld.config import SimConfig  # noqa: E402
from pim.environments.discworld.grid_target import categorical_target  # noqa: E402
from pim.environments.discworld.renderer import render_scene  # noqa: E402
from pim.environments.discworld.sim import Scene  # noqa: E402
from pim.figures.waterfall import DARK_BG  # noqa: E402

STD, BLINK, NRAY = "dw-noiseless", "dw-blink", ("dw-16ray", "dw-8ray", "dw-5ray")
T_STAR = 20            # the frame the frustum views show: the bench's edit frame, midway through the 40
ARROW_FRAMES = 15      # a velocity arrow spans this many frames of motion, from the disc's edge
N_GHOSTS = 7           # earlier frames drawn as fading discs behind the current one
MAX_HIDDEN = 3         # frames in which one disc may be fully occluded (crossing streaks need >= 1)
SEED = 0

# muted tones for the appearance cells: Okabe-Ito hues blended toward white (no meaning, only contrast)
_OI = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#0072B2")


def _tone(c: str, w: float) -> tuple:
    return tuple(w + (1 - w) * np.array(to_rgb(c)))


TONES = [_tone(c, 0.80) for c in _OI]
TONES_OWN = [_tone(c, 0.45) for c in _OI]                                 # the discs' own cells
CELL_EDGE = "#9c9c9c"


# ── data ────────────────────────────────────────────────────────────────────────────────
def read(inst: str) -> dict:
    """One instance's whole eval split (10k sequences) plus its sim contract."""
    with h5py.File(layout.eval_file("discworld", inst), "r") as f:
        d = {k: f[k][:] for k in ("positions", "velocities", "radii", "reflectivities", "colors",
                                  "obs_intensity", "obs_id", "obs_depth", "seeds")}
        d["visible"] = f["blink_visible"][:] if "blink_visible" in f else None
        d["sim"] = json.loads(f.attrs["config_json"])["dataset"]["sim"]
    return d


def seq(d: dict, i: int) -> dict:
    return {"index": int(i), "seed": int(d["seeds"][i]), "sim": d["sim"],
            "pos": d["positions"][i].astype(np.float64), "vel": d["velocities"][i].astype(np.float64),
            "radii": d["radii"][i].astype(np.float64), "refl": d["reflectivities"][i].astype(np.float64),
            "colors": d["colors"][i].astype(np.float64), "obs": d["obs_intensity"][i],
            "obs_id": d["obs_id"][i], "depth": d["obs_depth"][i],
            "visible": None if d["visible"] is None else d["visible"][i]}


def _runs(obs_id: np.ndarray):
    """(N, F, R) hit ids -> per disc: rays lit (N, F, 2), run centre (nan when unlit), first, last."""
    R = obs_id.shape[-1]
    k = np.arange(R)
    lit, cen, first, last = [], [], [], []
    for j in (0, 1):
        m = obs_id == j
        n = m.sum(-1)
        lit.append(n)
        cen.append(np.where(n > 0, (m * k).sum(-1) / np.maximum(n, 1), np.nan))
        first.append(m.argmax(-1))
        last.append(R - 1 - m[..., ::-1].argmax(-1))
    return tuple(np.stack(a, -1) for a in (lit, cen, first, last))


def pick(d: dict, rng: np.random.Generator) -> tuple[int, int]:
    """The selection rule (README): the two streaks cross (the discs swap lateral order between the
    first and last frame), with a disc fully occluded in at most MAX_HIDDEN frames; each run centre
    travels >= 0.2 R rays; at t* the discs are >= 2 depth units apart, their runs disjoint with a
    gap >= max(1, 0.05 R), each lit by >= max(1, 0.03 R) rays, and of different length."""
    lit, cen, first, last = _runs(d["obs_id"])
    R = d["obs_id"].shape[-1]
    always = (lit < 1).any(-1).sum(1) <= MAX_HIDDEN          # crossing streaks imply some full occlusion
    cross = np.sign(cen[:, 0, 0] - cen[:, 0, 1]) != np.sign(cen[:, -1, 0] - cen[:, -1, 1])
    travel = (np.nanmax(cen, 1) - np.nanmin(cen, 1)).min(1) >= 0.2 * R
    y = d["positions"][:, T_STAR, :, 1]
    depth = np.abs(y[:, 0] - y[:, 1]) >= 2.0
    f, la, n = first[:, T_STAR], last[:, T_STAR], lit[:, T_STAR]
    gap = np.maximum(f[:, 0] - la[:, 1], f[:, 1] - la[:, 0]) >= max(1, 0.05 * R)
    lit_ok = n.min(1) >= max(1, 0.03 * R)
    lengths = n[:, 0] != n[:, 1]
    ok = np.flatnonzero(always & cross & travel & depth & gap & lit_ok & lengths)
    return int(rng.choice(ok)), len(ok)


def pick_blink(d: dict, rng: np.random.Generator) -> tuple[int, int, dict]:
    """Blink rule (README): exactly one blackout in the sequence, of disc 1 (its marker ray is the
    rightmost, so marker and bracket share a side), 5-10 frames long, starting at frame >= 6 and
    ending by frame 33 (both markers present, context on both sides); the other disc lit in every
    frame, the blinking disc lit whenever visible and travelling >= 0.2 R rays."""
    v = d["visible"]
    N, F, _ = v.shape
    hidden = ~v
    n_hidden = hidden.sum(1)
    h = hidden[:, :, 1]
    one = ((n_hidden > 0).sum(1) == 1) & (n_hidden[:, 1] > 0)
    starts = (h[:, 1:] & ~h[:, :-1]).sum(1) + h[:, 0]
    a = h.argmax(1)
    b = F - 1 - h[:, ::-1].argmax(1)
    length = b - a + 1
    lit, cen, _, _ = _runs(d["obs_id"])
    R = d["obs_id"].shape[-1]
    other = (lit[:, :, 0] >= 1).all(1)
    self_lit = ((lit[:, :, 1] >= 1) | h).all(1)
    travel = (np.nanmax(cen[:, :, 1], 1) - np.nanmin(cen[:, :, 1], 1)) >= 0.2 * R
    ok = np.flatnonzero(one & (starts == 1) & (length >= 5) & (length <= 10) & (a >= 6) & (b <= F - 7)
                        & other & self_lit & travel)
    i = int(rng.choice(ok))
    return i, len(ok), {"disc": 1, "first_hidden": int(a[i]), "last_hidden": int(b[i]),
                        "markers": [int(a[i]) - 1, int(b[i])]}


def rerender(q: dict, sim: dict) -> np.ndarray:
    """The same world seen by a sibling instance: its own SimConfig, the canonical renderer."""
    cfg = dataclasses.replace(SimConfig(**sim), seed=q["seed"], n_objects=2)
    assert np.allclose(q["radii"], cfg.radius), "matched rendering needs the same disc radius"
    scene = Scene(positions=q["pos"], velocities=q["vel"], radii=q["radii"], colors=q["colors"],
                  reflectivities=q["refl"], config=cfg)
    return render_scene(scene)[2].astype(np.float32)


# ── drawing ─────────────────────────────────────────────────────────────────────────────
def kept_rays(sim: dict) -> np.ndarray:
    """Lateral parameter s of the KEPT rays: direction (s * x_far / y_far, 1), as render_frame casts
    them (uniform in s over obs_res rays), minus the two wall rays when drop_edge_rays (renderer._keep)."""
    s = np.linspace(-1.0, 1.0, int(sim["obs_res"]))
    return s[1:-1] if sim.get("drop_edge_rays") else s


def draw_frustum(ax, q: dict, t: int, *, dark=False, every=1, trail=True, arrows=True,
                 strip=False, cells=None, rays=True):
    sim = q["sim"]
    xn, yn, xf, yf = (float(sim[k]) for k in ("x_near", "y_near", "x_far", "y_far"))
    scale = xf / yf
    ink = "white" if dark else "black"
    ax.set_aspect("equal")
    ax.axis("off")
    top = yf + (1.55 if strip else 0.45)
    ax.set_xlim(-xf - 0.45, xf + 0.45)
    ax.set_ylim(-0.6, top)
    if dark:
        ax.add_patch(Rectangle((-xf - 0.45, -0.6), 2 * xf + 0.9, top + 0.6, facecolor=DARK_BG,
                               edgecolor=ps.FRAME, lw=0.5, zorder=0))
    if cells is not None:
        draw_cells(ax, q, t, cells)
    s = kept_rays(sim)
    ids, dep = q["obs_id"][t], q["depth"][t]
    if rays:
        for k in range(0, len(s), every):
            if ids[k] >= 0:          # a hit: the ray stops at the disc surface (obs_depth = its y)
                ax.plot([0, s[k] * scale * dep[k]], [0, dep[k]], color=ink, lw=0.5,
                        alpha=0.85 if dark else 0.7, zorder=3, solid_capstyle="butt")
            else:                    # a miss: through to the far plane
                ax.plot([0, s[k] * xf], [0, yf], color=ink, lw=0.3, alpha=0.3 if dark else 0.22, zorder=1)
    ax.add_patch(Polygon([(-xn, yn), (-xf, yf), (xf, yf), (xn, yn)], closed=True, fill=False,
                         edgecolor=ink, lw=0.8, zorder=4))
    ax.add_patch(Circle((0, 0), 0.22, facecolor=ink, edgecolor=ink, zorder=6))       # the observer
    for j in (0, 1):
        p, v, r, g = q["pos"][:, j], q["vel"][t, j], float(q["radii"][j]), float(q["refl"][j])
        if trail:                    # earlier frames as fading discs, up to the current one
            past = np.linspace(0, t, N_GHOSTS, endpoint=False).round().astype(int)
            for n, tau in enumerate(past):
                ax.add_patch(Circle(p[tau], r, facecolor=ink, edgecolor="none",
                                    alpha=0.05 + 0.30 * (n + 1) / len(past), zorder=2))
        ax.add_patch(Circle(p[t], r, facecolor=str(g), edgecolor=ink, lw=0.6, zorder=5))
        if arrows:
            u = v / np.linalg.norm(v)
            ax.annotate("", xy=p[t] + r * u + ARROW_FRAMES * v, xytext=p[t] + r * u, zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.8, mutation_scale=7,
                                        shrinkA=0, shrinkB=0))
    if strip:                        # frame t laid along the far plane: pixel k under ray k's far crossing
        ds = (s[1] - s[0]) * xf
        x0, x1, y0, h = s[0] * xf - ds / 2, s[-1] * xf + ds / 2, yf + 0.5, 0.75
        ax.imshow(q["obs"][t][None], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest",
                  extent=(x0, x1, y0, y0 + h), zorder=8)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, h, fill=False, edgecolor=ps.FRAME, lw=0.5, zorder=9))
        ax.text(xf + 0.7, y0 + h / 2, "$t^*$", ha="left", va="center", color="black")   # on the page, both modes


def draw_cells(ax, q: dict, t: int, target, n: int = 500):
    """The appearance partition over the REACHABLE region (centre one radius clear of every wall,
    sim.fully_in_frustum), coloured so that neighbouring cells differ; the discs' cells stronger."""
    sim = q["sim"]
    r, yn, yf, xf = (float(sim[k]) for k in ("radius", "y_near", "y_far", "x_far"))
    scale = xf / yf
    X, Y = np.meshgrid(np.linspace(-xf, xf, n), np.linspace(yn, yf, n))
    reach = (Y >= yn + r) & (Y <= yf - r) & (np.abs(X) <= scale * Y - r)
    lab = np.full(X.shape, -1)
    lab[reach] = target.cell_of(np.stack([X[reach], Y[reach]], -1), sim)
    G = int(lab.max()) + 1
    adj = [set() for _ in range(G)]
    for a, b in ((lab[:, 1:], lab[:, :-1]), (lab[1:, :], lab[:-1, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for u, w in set(zip(a[m].tolist(), b[m].tolist())):
            adj[u].add(w)
            adj[w].add(u)
    tone = np.full(G, -1)
    for c in range(G):
        tone[c] = next(k for k in range(len(TONES)) if k not in {tone[w] for w in adj[c]})
    own = set(target.cell_of(q["pos"][t], sim).tolist())
    for c in range(G):
        m = (lab == c).astype(float)
        ax.contourf(X, Y, m, levels=[0.5, 1.5], colors=[TONES_OWN[tone[c]] if c in own else TONES[tone[c]]],
                    zorder=0.5)
        ax.contour(X, Y, m, levels=[0.5], colors=[CELL_EDGE], linewidths=0.3, zorder=0.6)
    for c in own:
        ax.contour(X, Y, (lab == c).astype(float), levels=[0.5], colors=["black"], linewidths=0.9, zorder=3.5)


def draw_waterfall(ax, obs: np.ndarray, *, t_star=None, hidden=None, ticks=True, arrow=True):
    """A 40 x R observation sequence exactly as the canonical waterfall draws it, time downward."""
    F, R = obs.shape
    ax.imshow(obs, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor(ps.FRAME)
    ax.set_yticks([])
    if ticks:
        ax.set_xticks([0, R - 1])
        ax.set_xticklabels(["0", str(R - 1)], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.set_xlabel("ray", labelpad=0)
    else:
        ax.set_xticks([])
    if arrow:
        ax.annotate("", xy=(-0.055, 0.0), xytext=(-0.055, 1.0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=6,
                                    shrinkA=0, shrinkB=0), annotation_clip=False)
        ax.text(-0.10, 0.5, "$t$", transform=ax.transAxes, ha="right", va="center")
        ax.text(-0.10, 1.0 - 0.5 / F, "0", transform=ax.transAxes, ha="right", va="center", fontsize=7)
        ax.text(-0.10, 0.5 / F, str(F - 1), transform=ax.transAxes, ha="right", va="center", fontsize=7)
    row = lambda t: 1.0 - (t + 0.5) / F        # noqa: E731  axes-fraction y of a row's centre
    if t_star is not None:
        ax.plot([1.035], [row(t_star)], marker="<", ms=3.2, color="black", transform=ax.transAxes, clip_on=False)
        ax.text(1.075, row(t_star), "$t^*$", transform=ax.transAxes, ha="left", va="center")
    if hidden is not None:                     # the frames a disc is blacked out, bracketed on the right
        a, b = hidden
        y0, y1 = row(b) - 0.5 / F, row(a) + 0.5 / F
        x = 1.035
        ax.plot([x, x], [y0, y1], color="black", lw=0.7, transform=ax.transAxes, clip_on=False)
        for y in (y0, y1):
            ax.plot([x, x - 0.02], [y, y], color="black", lw=0.7, transform=ax.transAxes, clip_on=False)
        ax.text(x + 0.04, (y0 + y1) / 2, "hidden", rotation=90, transform=ax.transAxes,
                ha="left", va="center", fontsize=7.5)


def draw_strip(ax, frame: np.ndarray):
    ax.imshow(frame[None], cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor(ps.FRAME)


def letter(ax, s: str, dx=0.0):
    ax.text(dx, 1.0, s, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def piece(name: str, w: float, h: float, draw) -> None:
    """One exported element: a figure at its printed size (300 dpi rasters), drawn, saved beside this script."""
    f = plt.figure(figsize=(w, h), dpi=300)
    draw(f.add_subplot())
    ps.save(f, HERE / name)
    plt.close(f)


def draw_key(ax):
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    rows = [("disc, reflectivity 0.8", lambda y: ax.add_patch(Circle((0.8, y), 0.42, facecolor="0.8", edgecolor="black", lw=0.6))),
            ("disc, reflectivity 0.4", lambda y: ax.add_patch(Circle((0.8, y), 0.42, facecolor="0.4", edgecolor="black", lw=0.6))),
            ("velocity", lambda y: ax.annotate("", xy=(1.5, y), xytext=(0.1, y),
                                               arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=7))),
            ("earlier frames", lambda y: [ax.add_patch(Circle((0.35 + 0.3 * n, y), 0.3, facecolor="black", edgecolor="none",
                                                              alpha=0.05 + 0.3 * (n + 1) / 4)) for n in range(4)]),
            ("ray, hits a disc", lambda y: ax.plot([0.1, 1.5], [y, y], color="black", lw=0.5, alpha=0.7)),
            ("ray, no hit", lambda y: ax.plot([0.1, 1.5], [y, y], color="black", lw=0.3, alpha=0.22))]
    for n, (lab, draw) in enumerate(rows):
        draw(5.5 - n)
        ax.text(2.0, 5.5 - n, lab, va="center", ha="left", fontsize=8)


def composite(std, blk, span, eight, matched, app, *, dark: bool) -> None:
    """(a) standard frustum with frame t* along the far plane + its waterfall; (b) blink; (c) N-ray; (d) categorical."""
    f = plt.figure(figsize=(5.5, 4.3), dpi=300)
    gs = f.add_gridspec(2, 1, height_ratios=[2.5, 1.6], hspace=0.3, left=0.02, right=0.985, top=0.97, bottom=0.06)
    top = gs[0].subgridspec(1, 2, width_ratios=[0.86, 1.0], wspace=0.12)
    ax = f.add_subplot(top[0])
    draw_frustum(ax, std, T_STAR, dark=dark, strip=True)
    ax.set_anchor("W")
    letter(ax, "(a)", 0.02)
    draw_waterfall(f.add_subplot(top[1]), std["obs"], t_star=T_STAR)
    bot = gs[1].subgridspec(1, 4, width_ratios=[1.0, 1.25, 1.15, 1.25], wspace=0.42)
    ax = f.add_subplot(bot[0])
    draw_waterfall(ax, blk["obs"], hidden=(span["first_hidden"], span["last_hidden"]))
    letter(ax, "(b)", -0.2)
    ax = f.add_subplot(bot[1])
    draw_frustum(ax, eight, T_STAR, dark=dark)
    letter(ax, "(c)", 0.0)
    sub = bot[2].subgridspec(1, 3, wspace=0.3)
    for k, inst in enumerate(NRAY):
        ax = f.add_subplot(sub[k])
        draw_waterfall(ax, matched[inst], ticks=False, arrow=False)
        ax.set_title(f"{inst.split('-')[1][:-3]} rays", fontsize=7.5, pad=2)
    ax = f.add_subplot(bot[3])
    draw_frustum(ax, eight, T_STAR, cells=app, trail=False, arrows=False, dark=dark)
    letter(ax, "(d)", 0.0)
    ps.save(f, HERE / ("composite_dark" if dark else "composite"))
    plt.close(f)


# ── build ───────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    out = HERE
    rec = {"seed": SEED, "t_star": T_STAR, "arrow_frames": ARROW_FRAMES, "ghost_frames": N_GHOSTS,
           "ghost_frame_indices": np.linspace(0, T_STAR, N_GHOSTS, endpoint=False).round().astype(int).tolist(),
           "panels": {}}

    def record(name, q, n_ok, **extra):
        rec["panels"][name] = {"instance": name, "sequence_index": q["index"], "sequence_seed": q["seed"],
                               "n_rule_survivors": n_ok, "radii": q["radii"].tolist(),
                               "reflectivities": q["refl"].tolist(), "rays": int(q["obs"].shape[1]),
                               "positions_t_star": q["pos"][T_STAR].round(3).tolist(),
                               "velocities_t_star": q["vel"][T_STAR].round(4).tolist(), **extra}

    # (a) standard
    d = read(STD)
    i, n_ok = pick(d, rng)
    std = seq(d, i)
    record(STD, std, n_ok)
    print(f"standard  {STD}  seq {i} (seed {std['seed']}), {n_ok} sequences satisfy the rule")
    for dark in (False, True):
        tag = "dark" if dark else "light"
        piece(f"standard_frustum_{tag}", 2.7, 2.75, lambda ax: draw_frustum(ax, std, T_STAR, dark=dark))
        piece(f"standard_frustum_{tag}_strip", 2.7, 3.0, lambda ax: draw_frustum(ax, std, T_STAR, dark=dark, strip=True))
    piece("standard_frustum_light_every4", 2.7, 2.75, lambda ax: draw_frustum(ax, std, T_STAR, every=4))
    piece("standard_waterfall", 2.7, 2.75, lambda ax: draw_waterfall(ax, std["obs"], t_star=T_STAR))
    piece("standard_frame_tstar", 2.7, 0.22, lambda ax: draw_strip(ax, std["obs"][T_STAR]))
    del d

    # (b) blink
    d = read(BLINK)
    i, n_ok, span = pick_blink(d, rng)
    blk = seq(d, i)
    record(BLINK, blk, n_ok, blackout=span)
    print(f"blink     {BLINK}  seq {i} (seed {blk['seed']}), {n_ok} satisfy the rule; hidden frames {span}")
    piece("blink_waterfall", 2.7, 2.75,
          lambda ax: draw_waterfall(ax, blk["obs"], hidden=(span["first_hidden"], span["last_hidden"])))
    del d

    # (c) N-ray: one dw-8ray sequence, seen through 16 / 8 / 5 rays (matched), plus each instance's own draw
    sims = {}
    own = {}
    for inst in NRAY:
        d = read(inst)
        sims[inst] = d["sim"]
        i, n_ok = pick(d, rng)
        own[inst] = seq(d, i)
        record(inst, own[inst], n_ok)
        print(f"N-ray     {inst}  seq {i} (seed {own[inst]['seed']}), {n_ok} satisfy the rule")
        del d
    eight = own["dw-8ray"]
    matched = {inst: rerender(eight, sims[inst]) for inst in NRAY}
    assert np.array_equal(matched["dw-8ray"], eight["obs"]), "the canonical renderer must reproduce the stored 8-ray frames"
    rec["matched_nray"] = {"source": "dw-8ray", "sequence_index": eight["index"], "sequence_seed": eight["seed"],
                           "note": "positions of the dw-8ray sequence rendered by renderer.render_scene under the "
                                   "dw-16ray / dw-5ray SimConfig (radius 1.0 on all three); the 8-ray re-render "
                                   "equals the stored frames exactly"}
    piece("nray_frustum_8ray", 1.6, 1.7, lambda ax: draw_frustum(ax, eight, T_STAR))
    piece("nray_frustum_8ray_strip", 1.6, 1.9, lambda ax: draw_frustum(ax, eight, T_STAR, strip=True))
    for inst in NRAY:
        n = inst.split("-")[1]
        piece(f"nray_waterfall_{n}_matched", 0.75, 1.7, lambda ax: draw_waterfall(ax, matched[inst], ticks=False, arrow=False))
        piece(f"nray_waterfall_{n}_own", 0.75, 1.7, lambda ax: draw_waterfall(ax, own[inst]["obs"], ticks=False, arrow=False))

    # (d) categorical: the appearance partition on dw-8ray, the same sequence
    app = categorical_target("appearance")
    rec["categorical"] = {"target": "appearance (the partition the factorised appearance-fac target reads as "
                                    "run centre x run length)", "instance": "dw-8ray", "sequence_index": eight["index"],
                          "n_cells": int(app.n_cells(sims["dw-8ray"])),
                          "cells_at_t_star": app.cell_of(eight["pos"][T_STAR], sims["dw-8ray"]).tolist()}
    print(f"categorical  dw-8ray  {rec['categorical']['n_cells']} cells; discs at t* in cells {rec['categorical']['cells_at_t_star']}")
    piece("categorical_frustum_8ray", 1.6, 1.7,
          lambda ax: draw_frustum(ax, eight, T_STAR, cells=app, trail=False, arrows=False))
    piece("categorical_frustum_8ray_norays", 1.6, 1.7,
          lambda ax: draw_frustum(ax, eight, T_STAR, cells=app, trail=False, arrows=False, rays=False))

    piece("key_frustum", 1.5, 0.9, draw_key)
    for dark in (False, True):
        composite(std, blk, span, eight, matched, app, dark=dark)

    rec["rules"] = {"standard_and_nray": " ".join(pick.__doc__.split()), "blink": " ".join(pick_blink.__doc__.split()),
                    "max_hidden_frames": MAX_HIDDEN, "draw": "numpy default_rng(SEED).choice over the survivors, in the order "
                    "standard, blink, 16-ray, 8-ray, 5-ray"}
    json.dump(rec, open(out / "selection.json", "w"), indent=1)
    print("->", out / "selection.json")


if __name__ == "__main__":
    main()
