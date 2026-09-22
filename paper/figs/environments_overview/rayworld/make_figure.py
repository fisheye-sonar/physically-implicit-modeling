"""Rayworld and its variants: the environment figure pieces (paper ``fig:rayworld_and_variants``).

Real held-out sequences (``datasets/discworld/<inst>/eval/test.h5``) drawn with the canonical
geometry: ray directions exactly as ``renderer.render_frame`` spreads them (uniform in the
tangent of the viewing angle, wall rays dropped when the instance says so), hits and hit depths
from the stored ``obs_id`` / ``obs_depth``, appearance cells from
``grid_target.categorical_target("appearance")``, the matched N-ray strips from
``renderer.render_scene`` under each sibling instance's own ``SimConfig``. No metric, no
re-implemented rendering. CPU only. Two composites land beside this script — ``composite`` (two bands)
and ``composite_onerow`` (one band) — with their per-element exports in ``pieces/`` and ``pieces_onerow/``
(round 4, 2026-09-22); ``--all`` also regenerates the variants pruned in round 3.

    .pim/bin/python paper/figs/environments_overview/rayworld/make_figure.py [--all]
"""
from __future__ import annotations

import argparse
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
from matplotlib.colors import hsv_to_rgb  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

from pim.environments import layout  # noqa: E402
from pim.environments.discworld.config import SimConfig  # noqa: E402
from pim.environments.discworld.grid_target import categorical_target  # noqa: E402
from pim.environments.discworld.renderer import render_scene  # noqa: E402
from pim.environments.discworld.sim import Scene  # noqa: E402
from pim.figures.waterfall import DARK_BG, EDIT_LINE  # noqa: E402

PIECES, PIECES_ONEROW = HERE / "pieces", HERE / "pieces_onerow"
STD, BLINK, NRAY = "dw-noiseless", "dw-blink", ("dw-16ray", "dw-8ray", "dw-5ray")
T_STAR = 20            # the frame the frustum views show: the bench's edit frame, midway through the 40
ARROW_FRAMES = 15      # a velocity arrow spans this many frames of motion, from the disc's edge
N_GHOSTS = 7           # earlier frames drawn as fading discs behind the current one
MAX_HIDDEN = 3         # frames in which one disc may be fully occluded (crossing streaks need >= 1)
EVERY = 3              # the 128-ray frustum draws every EVERY-th ray (43 of 128): visual clarity only
ONEROW_EVERY = 5       # the one-row variant's frustum is small: fewer rays (26 of 128), drawn stronger
# The one-row band, in inches: panel sizes and the gaps between them (``onerow_geometry`` lays them out).
# Its labels are placed in POINTS off the panel edge, not in axes fractions, so they keep their clearance
# on panels this narrow.
ONEROW = {"band": 1.32, "wf_a": 0.56, "wf_b": 0.50, "strip": 0.30, "strip_gap": 0.105, "axis": 0.27,
          "gaps": (0.34, 0.46, 0.28, 0.11)}   # frustum→(a), (a)→(b), (b)→(c), (c)→(d)
DISC_ALPHA_CELLS = 0.65  # discs on the partition panel let the cells show through
SEED = 0
CELL_EDGE = "#9c9c9c"
CELL_STATS: dict = {}  # filled by draw_cells: the MEASURED colour separation of page-adjacent cells


def cell_adjacency(lab: np.ndarray, n: int) -> list[set]:
    """Which cells touch on the page, off the drawn label grid (4-neighbourhood)."""
    adj = [set() for _ in range(n)]
    for a, b in ((lab[:, 1:], lab[:, :-1]), (lab[1:, :], lab[:-1, :])):
        m = (a != b) & (a >= 0) & (b >= 0)
        for u, w in set(zip(a[m].tolist(), b[m].tolist())):
            adj[u].add(w)
            adj[w].add(u)
    return adj


def cell_colours(adj: list[set]) -> tuple[np.ndarray, np.ndarray]:
    """One colour per appearance cell, placed so that cells ADJACENT ON THE PAGE differ as strongly as
    the cell count allows. No sampling and no seed: the hues are the ``n`` evenly spaced points of the
    colour circle, and the cells take them greedily — most constrained cell first, ties by cell index —
    each taking the free hue whose circular distance to its already coloured neighbours is the largest.
    Saturation and value then cycle over four and three fixed levels to separate cells whose hues tie.
    Every colour stays light (high value, low saturation) so the rays, the discs and the centre dots
    read on top. Returns (light, strong): the strong tone is the same hue further saturated, for the
    two cells the discs are in.
    """
    n = len(adj)
    slot, taken = np.full(n, -1), np.zeros(n, bool)
    circ = lambda a, b: min(abs(a - b), n - abs(a - b))                    # noqa: E731  distance in slots
    for c in sorted(range(n), key=lambda c: (-len(adj[c]), c)):
        free = np.flatnonzero(~taken)
        near = [int(slot[w]) for w in adj[c] if slot[w] >= 0]
        h = int(free[np.argmax([min(circ(int(k), j) for j in near) for k in free])] if near else free[0])
        slot[c], taken[h] = h, True
    k = np.arange(n)
    hue = slot / n
    s = 0.18 + 0.13 * (k % 4) / 3
    v = 1.0 - 0.07 * (k % 3) / 2
    return (hsv_to_rgb(np.stack([hue, s, v], -1)),
            hsv_to_rgb(np.stack([hue, np.clip(s * 2.6, 0.0, 0.72), v * 0.94], -1)))


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


def frustum_extent(sim: dict, *, strip: bool = False, crop: bool = False) -> tuple[tuple, tuple]:
    """Data limits of a frustum view: the frustum plus a margin, room for the far-plane strip above it,
    and (``crop``) cut just below the near plane, observer and lower fan left out."""
    xf, yn, yf = (float(sim[k]) for k in ("x_far", "y_near", "y_far"))
    return (-xf - 0.45, xf + 0.45), (yn - 0.35 if crop else -0.6, yf + (1.55 if strip else 0.45))


def frustum_width(h: float, sim: dict, **kw) -> float:
    """Width (inches) a frustum view of height ``h`` inches needs at equal aspect."""
    (x0, x1), (y0, y1) = frustum_extent(sim, **kw)
    return h * (x1 - x0) / (y1 - y0)


def draw_frustum(ax, q: dict, t: int, *, dark=False, every=1, trail=True, arrows=True, strip=False,
                 cells=None, rays=True, crop=False, disc_alpha=1.0, strong=False):
    sim = q["sim"]
    xn, yn, xf, yf = (float(sim[k]) for k in ("x_near", "y_near", "x_far", "y_far"))
    scale = xf / yf
    ink = "white" if dark else "black"
    (x0, x1), (y0, y1) = frustum_extent(sim, strip=strip, crop=crop)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if dark:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=DARK_BG, edgecolor=ps.FRAME, lw=0.5, zorder=0))
    if cells is not None:
        draw_cells(ax, q, t, cells)
    s = kept_rays(sim)
    ids, dep = q["obs_id"][t], q["depth"][t]
    ys = yn if crop else 0.0                    # rays start at the observer, or at the near plane when cropped
    # (lw, alpha) for a ray that hits and one that misses; ``strong`` is the small one-row panel, where
    # the default weights disappear at print size
    hit_w, miss_w = ((0.55, 0.85), (0.36, 0.42)) if strong else ((0.5, 0.85 if dark else 0.7),
                                                                 (0.3, 0.3 if dark else 0.22))
    if rays:
        for k in range(0, len(s), every):
            if ids[k] >= 0:          # a hit: the ray stops at the disc surface (obs_depth = its y)
                ax.plot([s[k] * scale * ys, s[k] * scale * dep[k]], [ys, dep[k]], color=ink, lw=hit_w[0],
                        alpha=hit_w[1], zorder=3, solid_capstyle="butt")
            else:                    # a miss: through to the far plane
                ax.plot([s[k] * scale * ys, s[k] * xf], [ys, yf], color=ink, lw=miss_w[0],
                        alpha=miss_w[1], zorder=1)
    ax.add_patch(Polygon([(-xn, yn), (-xf, yf), (xf, yf), (xn, yn)], closed=True, fill=False,
                         edgecolor=ink, lw=0.8, zorder=4))
    if not crop:
        ax.add_patch(Circle((0, 0), 0.22, facecolor=ink, edgecolor=ink, zorder=6))   # the observer
    for j in (0, 1):
        p, v, r, g = q["pos"][:, j], q["vel"][t, j], float(q["radii"][j]), float(q["refl"][j])
        if trail:                    # earlier frames as fading discs, up to the current one
            past = np.linspace(0, t, N_GHOSTS, endpoint=False).round().astype(int)
            for n, tau in enumerate(past):
                ax.add_patch(Circle(p[tau], r, facecolor=ink, edgecolor="none",
                                    alpha=0.05 + 0.30 * (n + 1) / len(past), zorder=2))
        ax.add_patch(Circle(p[t], r, facecolor=str(g), edgecolor=ink, lw=0.6, alpha=disc_alpha, zorder=5))
        if cells is not None:        # the disc's actual centre, on top of the translucent disc
            ax.plot(*p[t], ls="none", marker="o", ms=2.0, color=ink, zorder=8)
        if arrows:
            u = v / np.linalg.norm(v)
            ax.annotate("", xy=p[t] + r * u + ARROW_FRAMES * v, xytext=p[t] + r * u, zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color=ink, lw=0.8, mutation_scale=7,
                                        shrinkA=0, shrinkB=0))
    if strip:                        # frame t laid along the far plane: pixel k under ray k's far crossing
        ds = (s[1] - s[0]) * xf
        sx0, sx1, sy0, h = s[0] * xf - ds / 2, s[-1] * xf + ds / 2, yf + 0.5, 0.75
        ax.imshow(q["obs"][t][None], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest",
                  extent=(sx0, sx1, sy0, sy0 + h), zorder=8)
        ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, h, fill=False, edgecolor=ps.FRAME, lw=0.5, zorder=9))
        ax.text(xf + 0.7, sy0 + h / 2, "$t^*$", ha="left", va="center", color="black")   # on the page, both modes


def _off(ax, dx: float = 0.0, dy: float = 0.0):
    """``ax.transAxes`` shifted by (dx, dy) POINTS: spacing that does not shrink with the panel."""
    return offset_copy(ax.transAxes, fig=ax.figure, x=dx, y=dy, units="points")


def draw_cells(ax, q: dict, t: int, target, n: int = 500):
    """The appearance partition over the REACHABLE region (centre one radius clear of every wall,
    sim.fully_in_frustum), one ``cell_colours`` colour per cell; the two discs' own cells stronger
    and outlined in black."""
    sim = q["sim"]
    r, yn, yf, xf = (float(sim[k]) for k in ("radius", "y_near", "y_far", "x_far"))
    scale = xf / yf
    X, Y = np.meshgrid(np.linspace(-xf, xf, n), np.linspace(yn, yf, n))
    reach = (Y >= yn + r) & (Y <= yf - r) & (np.abs(X) <= scale * Y - r)
    lab = np.full(X.shape, -1)
    lab[reach] = target.cell_of(np.stack([X[reach], Y[reach]], -1), sim)
    G = int(lab.max()) + 1
    adj = cell_adjacency(lab, G)
    light, strong = cell_colours(adj)
    pairs = {(min(u, w), max(u, w)) for u in range(G) for w in adj[u]}
    d = sorted(float(np.linalg.norm(light[u] - light[w])) for u, w in pairs)
    CELL_STATS.update(n_cells=G, n_adjacent_pairs=len(pairs), min_distance=round(d[0], 3),
                      median_distance=round(float(np.median(d)), 3))
    own = set(target.cell_of(q["pos"][t], sim).tolist())
    for c in range(G):
        m = (lab == c).astype(float)
        ax.contourf(X, Y, m, levels=[0.5, 1.5], colors=[strong[c] if c in own else light[c]], zorder=0.5)
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
    ax.set_xticks([])
    if ticks == "compact":     # the range just under the panel, the axis name centred on its own baseline
        for xx, lab, ha in ((0.0, "0", "left"), (1.0, str(R - 1), "right")):
            ax.text(xx, 0.0, lab, transform=_off(ax, 0, -2), ha=ha, va="top", fontsize=7)
        ax.text(0.5, 0.0, "ray", transform=_off(ax, 0, -11), ha="center", va="top", fontsize=8)
    elif ticks:
        ax.set_xticks([0, R - 1])
        ax.set_xticklabels(["0", str(R - 1)], fontsize=7)
        ax.tick_params(axis="x", length=2, pad=1.5)
        ax.set_xlabel("ray", labelpad=0)
    if arrow:
        ax.annotate("", xy=(-0.055, 0.0), xytext=(-0.055, 1.0), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="black", mutation_scale=6,
                                    shrinkA=0, shrinkB=0), annotation_clip=False)
        ax.text(-0.10, 0.5, "$t$", transform=ax.transAxes, ha="right", va="center")
        ax.text(-0.10, 1.0 - 0.5 / F, "0", transform=ax.transAxes, ha="right", va="center", fontsize=7)
        ax.text(-0.10, 0.5 / F, str(F - 1), transform=ax.transAxes, ha="right", va="center", fontsize=7)
    row = lambda t: 1.0 - (t + 0.5) / F        # noqa: E731  axes-fraction y of a row's centre
    # On the compact (one-row) panels the caret, the bracket and their labels are offset in POINTS, so
    # they keep their clearance from the panel edge and from each other; the wide panels keep fractions.
    comp = ticks == "compact"
    if t_star is not None:                     # the pointer, and a thin light outline around the row itself
        ax.plot([1.0 if comp else 1.035], [row(t_star)], marker="<", ms=3.2, color="black", clip_on=False,
                transform=_off(ax, 5) if comp else ax.transAxes)
        ax.text(1.0 if comp else 1.075, row(t_star), "$t^*$", ha="left", va="center",
                transform=_off(ax, 11) if comp else ax.transAxes)
        ax.add_patch(Rectangle((-0.5, t_star - 0.5), R, 1.0, fill=False, edgecolor=EDIT_LINE, lw=0.6, zorder=3))
    if hidden is not None:                     # the frames a disc is blacked out, bracketed on the right
        a, b = hidden
        y0, y1 = row(b) - 0.5 / F, row(a) + 0.5 / F
        tr = _off(ax, 4) if comp else ax.transAxes
        x, d = (1.0, 0.05) if comp else (1.035, 0.02)
        ax.plot([x, x], [y0, y1], color="black", lw=0.7, transform=tr, clip_on=False)
        for y in (y0, y1):
            ax.plot([x, x - d], [y, y], color="black", lw=0.7, transform=tr, clip_on=False)
        ax.text(1.0 if comp else 1.075, (y0 + y1) / 2, "hidden", rotation=90, ha="left", va="center",
                fontsize=7.5, transform=_off(ax, 9) if comp else ax.transAxes)  # centred on the bracket


def draw_strip(ax, frame: np.ndarray):
    ax.imshow(frame[None], cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor(ps.FRAME)


def piece(name: str, w: float, h: float, draw, into: Path = PIECES) -> None:
    """One exported element: a figure at its printed size (300 dpi rasters), drawn, saved into ``into``."""
    f = plt.figure(figsize=(w, h), dpi=300)
    draw(f.add_axes([0, 0, 1, 1]))          # the axes fill the figure: a strip piece IS its nominal size
    ps.save(f, into / name)
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


# ── composites: axes placed in inches from the page's bottom-left, letters in their own margin ──
def _ax(f, x: float, y: float, w: float, h: float):
    W, H = f.get_size_inches()
    return f.add_axes([x / W, y / H, w / W, h / H])


def _letter(f, x: float, y: float, s: str) -> None:
    W, H = f.get_size_inches()
    f.text(x / W, y / H, s, fontweight="bold", ha="left", va="bottom")


def _strips(f, x: float, y: float, w: float, gap: float, h: float, matched: dict, size: float = 7.5,
            below: bool = False) -> None:
    """The three matched N-ray strips side by side, labelled above (or BELOW, where the band's letter
    row is needed for the panel letter)."""
    for k, inst in enumerate(NRAY):
        ax = _ax(f, x + k * (w + gap), y, w, h)
        draw_waterfall(ax, matched[inst], ticks=False, arrow=False)
        lab = f"{inst.split('-')[1][:-3]} rays"
        if below:   # on the one-row band's lower baseline, level with the waterfalls' "ray"
            ax.text(0.5, 0.0, lab, transform=_off(ax, 0, -11), ha="center", va="top", fontsize=size)
        else:
            ax.set_title(lab, fontsize=size, pad=2)


def composite(std, blk, span, eight, matched, app) -> None:
    """THE composite (5.5 in wide): (a) frustum with the t* strip + its waterfall and (b) blink across the top;
    (c) the 16/8/5-ray strips and (d) the cells below. Panels placed in inches, letters in a thin margin
    above each band; the frustum's size follows the top band's height (equal aspect)."""
    hb, hf = 1.00, 1.80                                    # bottom band; frustum (= top band) height
    fb = hb + 0.30                                         # the frustum's bottom: the (c)(d) letters sit in between
    f = plt.figure(figsize=(5.5, fb + hf + 0.2), dpi=300)
    wf = frustum_width(hf, std["sim"], strip=True)
    draw_frustum(_ax(f, 0.02, fb, wf, hf), std, T_STAR, every=EVERY, strip=True)
    _letter(f, 0.02, fb + hf + 0.04, "(a)")
    x = 0.02 + wf + 0.44                                   # past the strip's t* label and the waterfall's t labels
    draw_waterfall(_ax(f, x, fb + 0.30, 1.55, hf - 0.30), std["obs"], t_star=T_STAR)
    x += 1.55 + 0.50                                       # the t* pointer, then (b)'s own t labels
    draw_waterfall(_ax(f, x, fb + 0.30, 0.95, hf - 0.30), blk["obs"],
                   hidden=(span["first_hidden"], span["last_hidden"]))
    _letter(f, x - 0.21, fb + hf + 0.04, "(b)")
    _letter(f, 0.02, hb + 0.04, "(c)")
    _strips(f, 0.26, 0.0, 1.15, 0.10, hb, matched)
    wd = frustum_width(hb, eight["sim"], crop=True)
    draw_frustum(_ax(f, 5.48 - wd, 0.0, wd, hb), eight, T_STAR, cells=app, trail=False, arrows=False,
                 crop=True, disc_alpha=DISC_ALPHA_CELLS)
    _letter(f, 5.48 - wd, hb + 0.04, "(d)")
    ps.save(f, HERE / "composite")
    plt.close(f)


def onerow_geometry(std_sim: dict, eight_sim: dict) -> dict:
    """The one-row band's panel sizes and x positions, in inches — ONE place, so the composite and the
    pieces_onerow exports cannot drift apart. (d) takes what is left after the gaps the labels need."""
    g1, g2, g3, g4 = ONEROW["gaps"]
    g = {"band": ONEROW["band"], "axis": ONEROW["axis"],
         "wf": frustum_width(ONEROW["band"], std_sim, strip=True)}
    x = 0.02 + g["wf"] + g1                                  # past the strip's t* label
    g["x_wa"] = x
    x += ONEROW["wf_a"] + g2                                 # past (a)'s t* caret and label, then (b)'s t labels
    g["x_wb"] = x
    x += ONEROW["wf_b"] + g3                                 # past the hidden bracket and its label
    g["x_strips"] = x
    x += 3 * ONEROW["strip"] + 2 * ONEROW["strip_gap"] + g4
    g["x_d"] = x
    g["wd"] = 5.48 - x
    g["hd"] = g["wd"] / frustum_width(1.0, eight_sim, crop=True)
    return g


def composite_onerow(std, blk, span, eight, matched, app) -> None:
    """The ONE-ROW alternative (5.5 in wide, about 1.85 in tall): (a) frustum with the t* strip and its
    waterfall, (b) blink, (c) the 16/8/5-ray strips, (d) the cells — a single band, the letters in a
    strip above it. Sizes and gaps come from ``onerow_geometry``; the waterfalls carry a compact two
    baseline axis (the range under the panel, "ray" centred below it), the markers are spaced in points
    rather than axes fractions, and (a) draws ``ONEROW_EVERY``-th ray at heavier weights, because the
    two-band composite's thin pale rays vanish at this size."""
    g = onerow_geometry(std["sim"], eight["sim"])
    hb, xlab = g["band"], g["axis"]
    top = xlab + hb                        # the band's top edge
    f = plt.figure(figsize=(5.5, top + 0.26), dpi=300)
    ytxt = top + 0.10                      # the letters sit clear of the artwork
    draw_frustum(_ax(f, 0.02, xlab, g["wf"], hb), std, T_STAR, every=ONEROW_EVERY, strip=True, strong=True)
    _letter(f, 0.02, ytxt, "(a)")
    draw_waterfall(_ax(f, g["x_wa"], xlab, ONEROW["wf_a"], hb), std["obs"], t_star=T_STAR, ticks="compact")
    draw_waterfall(_ax(f, g["x_wb"], xlab, ONEROW["wf_b"], hb), blk["obs"], ticks="compact",
                   hidden=(span["first_hidden"], span["last_hidden"]))
    _letter(f, g["x_wb"] - 0.16, ytxt, "(b)")
    _letter(f, g["x_strips"] - 0.02, ytxt, "(c)")   # the ray counts go BELOW the strips, so this row is free
    _strips(f, g["x_strips"], xlab, ONEROW["strip"], ONEROW["strip_gap"], hb, matched, size=7.0, below=True)
    draw_frustum(_ax(f, g["x_d"], xlab + (hb - g["hd"]) / 2, g["wd"], g["hd"]), eight, T_STAR, cells=app,
                 trail=False, arrows=False, crop=True, disc_alpha=DISC_ALPHA_CELLS)
    _letter(f, g["x_d"], ytxt, "(d)")
    ps.save(f, HERE / "composite_onerow")
    plt.close(f)


def composite_rows(std, blk, span, eight, matched, app) -> None:
    """The alternative layout (--all, pieces/composite_rows): (a) frustum + waterfall across the top;
    (b) blink | (c) 16/8/5 rays | (d) cells below; 5.5 x 3.0 in."""
    f = plt.figure(figsize=(5.5, 3.0), dpi=300)
    top, bot = 2.77, 1.30                                  # the top band's panels; letters at 2.84
    wf = frustum_width(top - bot, std["sim"], strip=True)
    draw_frustum(_ax(f, 0.02, bot, wf, top - bot), std, T_STAR, every=EVERY, strip=True)
    _letter(f, 0.02, 2.84, "(a)")
    draw_waterfall(_ax(f, 0.02 + wf + 0.64, bot + 0.30, 3.0, top - bot - 0.30), std["obs"], t_star=T_STAR)
    hb = 1.08                                              # the bottom band's panels; letters at 1.15
    draw_waterfall(_ax(f, 0.24, 0.0, 0.90, hb), blk["obs"], hidden=(span["first_hidden"], span["last_hidden"]),
                   ticks=False)
    _letter(f, 0.02, 1.15, "(b)")
    _letter(f, 1.58, 1.15, "(c)")
    _strips(f, 1.82, 0.0, 0.62, 0.08, hb, matched)
    wd = frustum_width(hb, eight["sim"], crop=True)
    draw_frustum(_ax(f, 5.48 - wd, 0.0, wd, hb), eight, T_STAR, cells=app, trail=False, arrows=False,
                 crop=True, disc_alpha=DISC_ALPHA_CELLS)
    _letter(f, 5.48 - wd, 1.15, "(d)")
    ps.save(f, PIECES / "composite_rows")
    plt.close(f)


# ── build ───────────────────────────────────────────────────────────────────────────────
def main(all_: bool = False):
    """The kept set by default; ``all_`` adds the variants pruned in round 3 (every 2nd / 4th ray, dark
    frustums, the _own strips, the no-rays partition, the rows composite), all into pieces/."""
    rng = np.random.default_rng(SEED)
    out = HERE
    rec = {"seed": SEED, "t_star": T_STAR, "arrow_frames": ARROW_FRAMES, "ghost_frames": N_GHOSTS,
           "ghost_frame_indices": np.linspace(0, T_STAR, N_GHOSTS, endpoint=False).round().astype(int).tolist(),
           "rays_drawn_every": EVERY, "disc_alpha_on_cells": DISC_ALPHA_CELLS, "panels": {}}

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
    for every in ((1, 2, 3, 4) if all_ else (1, EVERY)):
        tag = "" if every == 1 else f"_every{every}"
        piece(f"standard_frustum_light{tag}", 2.7, 2.75, lambda ax: draw_frustum(ax, std, T_STAR, every=every))
        piece(f"standard_frustum_light{tag}_strip", 2.7, 3.0,
              lambda ax: draw_frustum(ax, std, T_STAR, every=every, strip=True))
    for every in ((1, EVERY) if all_ else ()):
        tag = "" if every == 1 else f"_every{every}"
        piece(f"standard_frustum_dark{tag}", 2.7, 2.75, lambda ax: draw_frustum(ax, std, T_STAR, every=every, dark=True))
        piece(f"standard_frustum_dark{tag}_strip", 2.7, 3.0,
              lambda ax: draw_frustum(ax, std, T_STAR, every=every, dark=True, strip=True))
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
        piece(f"nray_waterfall_{n}_matched", 1.25, 1.6, lambda ax: draw_waterfall(ax, matched[inst], ticks=False, arrow=False))
        if all_:
            piece(f"nray_waterfall_{n}_own", 1.25, 1.6, lambda ax: draw_waterfall(ax, own[inst]["obs"], ticks=False, arrow=False))

    # (d) categorical: the appearance partition on dw-8ray, the same sequence
    app = categorical_target("appearance")
    rec["categorical"] = {"target": "appearance (the partition the factorised appearance-fac target reads as "
                                    "run centre x run length)", "instance": "dw-8ray", "sequence_index": eight["index"],
                          "n_cells": int(app.n_cells(sims["dw-8ray"])),
                          "cells_at_t_star": app.cell_of(eight["pos"][T_STAR], sims["dw-8ray"]).tolist()}
    print(f"categorical  dw-8ray  {rec['categorical']['n_cells']} cells; discs at t* in cells {rec['categorical']['cells_at_t_star']}")
    cat = dict(cells=app, trail=False, arrows=False, disc_alpha=DISC_ALPHA_CELLS)
    piece("categorical_frustum_8ray", 2.2, 2.3, lambda ax: draw_frustum(ax, eight, T_STAR, **cat))
    piece("categorical_frustum_8ray_crop", 2.4, 2.4 / frustum_width(1.0, sims["dw-8ray"], crop=True),
          lambda ax: draw_frustum(ax, eight, T_STAR, crop=True, **cat))
    if all_:
        piece("categorical_frustum_8ray_norays", 2.2, 2.3, lambda ax: draw_frustum(ax, eight, T_STAR, rays=False, **cat))

    piece("key_frustum", 1.5, 0.9, draw_key)
    composite(std, blk, span, eight, matched, app)

    # the one-row alternative and ITS pieces, at the sizes that layout uses (pieces_onerow/)
    one = dict(into=PIECES_ONEROW)
    g = onerow_geometry(std["sim"], sims["dw-8ray"])   # the sizes the one-row composite places
    hb = g["band"]
    for every in (ONEROW_EVERY, 6):                    # the two ray densities asked for: 26 and 21 of 128
        piece(f"standard_frustum_light_every{every}_strip", g["wf"], hb,
              lambda ax: draw_frustum(ax, std, T_STAR, every=every, strip=True, strong=True), **one)
    piece("standard_waterfall", ONEROW["wf_a"], hb,
          lambda ax: draw_waterfall(ax, std["obs"], t_star=T_STAR, ticks="compact"), **one)
    piece("blink_waterfall", ONEROW["wf_b"], hb,
          lambda ax: draw_waterfall(ax, blk["obs"], hidden=(span["first_hidden"], span["last_hidden"]),
                                    ticks="compact"), **one)
    for inst in NRAY:
        piece(f"nray_waterfall_{inst.split('-')[1]}_matched", ONEROW["strip"], hb,
              lambda ax: draw_waterfall(ax, matched[inst], ticks=False, arrow=False), **one)
    piece("categorical_frustum_8ray_crop", g["wd"], g["hd"],
          lambda ax: draw_frustum(ax, eight, T_STAR, crop=True, **cat), **one)
    composite_onerow(std, blk, span, eight, matched, app)
    if all_:
        composite_rows(std, blk, span, eight, matched, app)

    rec["rules"] = {"standard_and_nray": " ".join(pick.__doc__.split()), "blink": " ".join(pick_blink.__doc__.split()),
                    "max_hidden_frames": MAX_HIDDEN, "draw": "numpy default_rng(SEED).choice over the survivors, in the order "
                    "standard, blink, 16-ray, 8-ray, 5-ray"}
    rec["outputs"] = {"composite": "composite.pdf/.png: (a) frustum with the t* strip + waterfall and (b) blink "
                                   "across the top, (c) 16/8/5-ray matched strips and (d) the cells below; "
                                   "5.5 x 3.3 in (two bands)",
                      "composite_onerow": f"composite_onerow.pdf/.png: the same four panels in ONE band, "
                                          f"5.5 x 1.79 in, its frustum drawing every {ONEROW_EVERY}-th ray "
                                          f"(heavier weights) and its waterfalls carrying a two baseline "
                                          f"compact axis; sizes and gaps in ONEROW / onerow_geometry, "
                                          f"pieces at those sizes in pieces_onerow/",
                      "onerow_panels_in": {"frustum": 1.20, "waterfall_a": ONEROW["wf_a"],
                                           "waterfall_b": ONEROW["wf_b"], "nray_strip": ONEROW["strip"],
                                           "nray_strip_gap": ONEROW["strip_gap"], "cells": 0.90,
                                           "band_height": ONEROW["band"]},
                      "pieces": sorted(p.stem for p in PIECES.glob("*.pdf")),
                      "pieces_onerow": sorted(p.stem for p in PIECES_ONEROW.glob("*.pdf")),
                      "pruned_2026-09-21": "every 2nd / 4th ray frustums, dark frustums, _own strips, the no-rays "
                                           "partition, composite_v1(_dark), composite_v2_rows: regenerate with --all "
                                           "(git history keeps the committed ones)"}
    rec["cell_colours"] = {**CELL_STATS, "deterministic": True,
                           "construction": " ".join(cell_colours.__doc__.split()).split("Returns")[0].strip()}
    rec["recommended"] = {"rays_drawn_every": EVERY, "rays_drawn_every_onerow": ONEROW_EVERY,
                          "nray_strips": "matched", "frustum_page": "light"}
    json.dump(rec, open(out / "selection.json", "w"), indent=1)
    print("->", out / "selection.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="also regenerate the pruned variants (every 2nd / 4th ray, dark frustums, _own strips, "
                         "no-rays partition, the rows composite) into pieces/")
    main(ap.parse_args().all)
