"""The omniscient-2D analogue of the waterfall — ONE definition, used by every
notebook in this thread.

Why this file exists
--------------------
`CLAUDE.md` makes a waterfall **mandatory** for any claim about an effect on the
generations, and fixes its spec precisely. That spec is written for a **1D**
observation: a frame is a row of pixels, so a whole rollout fits in one image with
time on the vertical axis. The omniscient observation is a 48x64 **2D raster**, so a
single frame already uses both image axes and a literal waterfall cannot be drawn.

This module is the adaptation. It keeps every *content* requirement of the spec and
changes only what the extra spatial dimension forces. It lives in the thread
directory rather than in each notebook because the spec's real instruction is "one
definition, do not re-implement per notebook" — importing one module is a stricter
reading of that than copying a helper into five notebooks.

What is preserved from the 1D spec (all of it, unchanged in meaning)
--------------------------------------------------------------------
* ``cmap="gray"`` on the dark background. Never magma / viridis / pink-purple.
* A **GT (sim clean-obs) reference** arm, always present, always first.
* **Noisy context frames** before the edit — the actual observations the model was
  teacher-forced on (``edits.obs``), NOT the clean render (only GT is clean) —
  separated from the post-edit frames by a marked **edit boundary**.
* Below/after that boundary, **every arm shows its OWN free-run starting at step 0**.
  No shared teacher-forced ``ef`` cell is ever painted across arms — that remains
  banned. The GT arm shows ``clean_obs[ef:ef+K]``.
* **Alignment:** ``warm_up_to_edit`` teacher-forces ``obs[0..ef-1]``, so a
  predict-next model's rollout **step 0 is frame ef** — ``ROLL[:, 0:K]`` is plotted
  against ``clean_obs[ef:ef+K]``, no slicing, no dropped step. An arm that was fed
  ``obs[ef]`` (First Obs. TF) **leads by one frame**; pass ``leads_by_one=True`` and
  it is labelled as such rather than the other arms being re-aligned to it.
* **Green = target**, **red dashed = ghost** locators.
* A **figure-top legend**, a **single what-is-shown title** (never a result), and the
  arm's **headline metric in its own label** so picture and number are read together.
* Sized so every cell stays full size — add arms by growing the figure, never by
  shrinking cells.
* Fixed ``vmin=0, vmax=1`` on every cell. Per-cell autoscaling would make a collapsed
  arm look normal, which is precisely the failure these panels exist to catch.

What necessarily changes, and why
----------------------------------
1. **Axes swap.** Arms become **rows** and time becomes **columns**. In 1D, arms were
   columns because time already owned the vertical axis of each image. Here each cell
   is a whole frame, so the grid's two axes are free; time-left-to-right is the
   reading order a viewer expects.

2. **Time is subsampled.** A 1D waterfall shows all ~21 frames because a frame costs
   one pixel row. Here a frame costs a whole cell, so showing 21 would either shrink
   cells below legibility or make the figure metres wide — both forbidden. The
   default keeps 3 context frames and 5 rollout steps (0, 3, 7, 11, 14), which
   preserves the two things the spec's frame series is *for*: what the model was fed,
   and whether the edit lands and then holds. Every displayed step is labelled with
   its true index, and `steps=` makes the choice explicit and auditable per figure.
   **`frame_trails` complements this** by compositing *every* step, so nothing is
   hidden by the subsample — pair them whenever the question is about persistence.

3. **Locators are circles, not vertical lines.** In 1D a position projects to a ray
   index, so a locator is a line. In 2D the target and ghost are *places*: they are
   drawn as circles of the true object radius at the true world coordinates, via
   ``imshow(extent=...)`` in world units.

Reading the panels
------------------
`frame_grid` shows raw model output and is the one that catches degradation — a
collapsed arm shows as noise or saturation in cells that should hold a clean disc.
`frame_trails` compresses each arm's whole rollout into one image and is the one that
shows *where the object went*. Report them together; an Edit Index that moved looks
identical in a scorecard whether the edit landed or the output degraded.

`frame_animation` is a third, **optional** view (added 2026-08-12 with the spec's
approval). It obeys `CLAUDE.md`'s animation rules — numbered persistent title, ~3 fps,
holds on the key frames — and is the most natural medium for a 2D raster. It is an
**addition, never a replacement**: a claim still ships with the grid + trails pair,
because a GIF cannot be read in a committed notebook diff or a paper.

Status
------
**Approved by Sevan 2026-08-12** and promoted: this is the sanctioned form for any
2D-raster observation, governed by `CLAUDE.md` § Waterfalls. Full spec and the record
of what was decided: `WATERFALL_SPEC_2D.md` beside this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from pim.simulator.config import SimConfig
from pim.simulator.render2d import grid_shape, unflatten, world_extent

# ── Fixed aesthetic (the dark simulator theme; see CLAUDE.md) ─────────────────
BG = "#0a0a14"
FG = "#a3adc2"
TARGET_C = "#009E73"  # green  — where the edited object must END UP
GHOST_C = "#D55E00"  # red    — where it was, and must VACATE
CMAP = "gray"
VMIN, VMAX = 0.0, 1.0

DEFAULT_CTX = 3
DEFAULT_STEPS = (0, 3, 7, 11, 14)

# World aspect (12 wide x 9 deep) -> every cell is 4:3. Fixed here rather than at the
# call site so a figure can only ever be resized by changing `cell`, never by
# squashing cells (which CLAUDE.md forbids: add arms by widening, never by shrinking).
CELL_ASPECT = 3.0 / 4.0


@dataclass
class Arm:
    """One column-family of the comparison: an editor, a reference, or GT.

    Attributes
    ----------
    name    : display label. Name it by its EDITOR, never by its edit site.
    roll    : (N, K, R) the arm's own free-run; **step 0 must decode frame `ef`**.
    metric  : headline metric string shown beneath the name (e.g. "Edit Index -0.44").
              Required for model arms so the picture and the number are read together;
              references may pass None.
    leads_by_one : True only for an arm fed `obs[ef]` (First Obs. TF). Labelled, never
              re-aligned.
    is_gt   : marks the ground-truth reference row (drawn first, styled apart).
    """

    name: str
    roll: np.ndarray
    metric: str | None = None
    leads_by_one: bool = False
    is_gt: bool = False


def _draw(ax, frame_flat, cfg):
    ax.imshow(
        unflatten(frame_flat, cfg),
        cmap=CMAP,
        vmin=VMIN,
        vmax=VMAX,
        origin="lower",  # row 0 is the NEAR plane; near at the bottom, as in the sim view
        extent=world_extent(cfg),
        # "equal", never "auto": the cell must carry the world's true 12x9 aspect, or
        # the locator circles render as ellipses and apparent object shape is a lie.
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a2f45")


def _locators(ax, target_xy, ghost_xy, radius):
    """Green = target (must end up here), red dashed = ghost (must vacate)."""
    if ghost_xy is not None:
        ax.add_patch(
            Circle(ghost_xy, radius, fill=False, ec=GHOST_C, lw=1.4, ls="--", zorder=5)
        )
    if target_xy is not None:
        ax.add_patch(
            Circle(target_xy, radius, fill=False, ec=TARGET_C, lw=1.6, zorder=6)
        )


def _legend(fig, extra=(), y=0.955):
    """Figure-top legend (required by the spec).

    `y` is passed explicitly because the legend must sit ABOVE the per-column time
    labels; leaving it at a fixed default let it overlap them in the first render.
    """
    handles = [
        Line2D(
            [],
            [],
            color=TARGET_C,
            lw=1.8,
            label="target — where the edited object must end up",
        ),
        Line2D(
            [],
            [],
            color=GHOST_C,
            lw=1.6,
            ls="--",
            label="ghost — where it was, and must vacate",
        ),
        *extra,
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        labelcolor=FG,
        fontsize=8.5,
    )


def frame_grid(
    arms: list[Arm],
    *,
    cfg: SimConfig,
    sample: int,
    ctx_obs: np.ndarray,  # (N, T, R) — the NOISY observations, edits.obs
    clean_obs: np.ndarray,  # (N, T, R) — clean render, for the GT arm only
    edit_frame: int,
    target_xy: np.ndarray | None = None,  # (2,) world coords
    ghost_xy: np.ndarray | None = None,  # (2,) world coords
    steps: tuple[int, ...] = DEFAULT_STEPS,
    n_ctx: int = DEFAULT_CTX,
    title: str = "",
    fig_num: str = "",
    cell: float = 1.6,
):
    """Arms x time grid of 2D frames — the omniscient-2D analogue of the waterfall.

    Every arm gets its own row: `n_ctx` noisy pre-edit context frames, an edit
    boundary, then that arm's OWN free-run at `steps`. The GT arm's post-edit cells
    are `clean_obs[ef + s]`.

    `title` states what is shown, never a result. `fig_num` is the figure number
    required by CLAUDE.md (e.g. "Fig 3").
    """
    ef = edit_frame
    ctx_idx = list(range(ef - n_ctx, ef))
    n_col = len(ctx_idx) + len(steps)
    n_row = len(arms)

    # Header space is allocated per-figure rather than as a fixed fraction: with a
    # fixed fraction the legend and the per-column time labels collide as soon as the
    # figure gets short (caught in the first render of this panel).
    head_in, foot_in = 1.35, 0.62
    fig_h = cell * CELL_ASPECT * n_row + head_in + foot_in
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(cell * n_col + 2.3, fig_h), squeeze=False
    )
    fig.patch.set_facecolor(BG)

    for r, arm in enumerate(arms):
        for c, t in enumerate(ctx_idx):
            ax = axes[r][c]
            ax.set_facecolor(BG)
            # The actual (noisy) frames the model was teacher-forced on -- NOT the
            # clean render. Only the GT arm is ever clean.
            _draw(ax, ctx_obs[sample, t], cfg)
            if r == 0:
                ax.set_title(f"t={t}", color=FG, fontsize=7.5, pad=3)

        for c, s in enumerate(steps):
            ax = axes[r][len(ctx_idx) + c]
            ax.set_facecolor(BG)
            if arm.is_gt:
                _draw(ax, clean_obs[sample, ef + s], cfg)
            else:
                _draw(ax, arm.roll[sample, s], cfg)
            _locators(ax, target_xy, ghost_xy, cfg.radius)
            if r == 0:
                # Neutral colour: green is reserved for the target locator, and using
                # it for a time label made the two read as the same thing.
                ax.set_title(f"step {s}  (t={ef + s})", color=FG, fontsize=7.5, pad=3)
            # The edit boundary: a bright edge on the first post-edit cell of every row.
            if c == 0:
                ax.spines["left"].set_color("#E69F00")
                ax.spines["left"].set_linewidth(2.4)

        lead = "  (leads by 1 frame)" if arm.leads_by_one else ""
        label = f"{arm.name}{lead}"
        if arm.metric:
            label += f"\n{arm.metric}"
        axes[r][0].set_ylabel(
            label,
            color="w" if arm.is_gt else FG,
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontweight="bold" if arm.is_gt else "normal",
        )

    top = 1.0 - head_in / fig_h
    _legend(
        fig,
        extra=[
            Line2D(
                [],
                [],
                color="#E69F00",
                lw=2.4,
                label="edit frame (first post-edit cell)",
            ),
        ],
        y=top + 0.052 * (head_in / fig_h) * 6,
    )
    head = f"{fig_num} — {title}" if fig_num else title
    fig.suptitle(head, color="w", fontsize=11, y=1.0 - 0.28 * head_in / fig_h)
    fig.text(
        0.5,
        0.20 * foot_in / fig_h,
        f"left of the orange edge: the noisy observations every arm was teacher-forced on "
        f"(t={ctx_idx[0]}..{ctx_idx[-1]}).  right: each arm's OWN free-run from step 0.  "
        f"GT arm shows the simulator's clean render.  gray, fixed scale 0–1.",
        color=FG,
        fontsize=7.5,
        ha="center",
    )
    fig.subplots_adjust(
        top=top,
        bottom=foot_in / fig_h,
        left=0.125,
        right=0.995,
        hspace=0.10,
        wspace=0.05,
    )
    return fig


def frame_trails(
    arms: list[Arm],
    *,
    cfg: SimConfig,
    sample: int,
    clean_obs: np.ndarray,
    edit_frame: int,
    n_steps: int = 15,
    target_xy: np.ndarray | None = None,
    ghost_xy: np.ndarray | None = None,
    title: str = "",
    fig_num: str = "",
    cell: float = 2.0,
    n_cols: int | None = None,
):
    """One cell per arm: the arm's WHOLE rollout composited into a single frame.

    Every step is included — this is the companion that guarantees `frame_grid`'s
    time subsample hides nothing. Later steps are drawn brighter, so the composite
    reads as a motion trail: a landed edit shows a trail arriving inside the green
    circle, a failed one a trail sitting on the red dashed circle, a collapsed one a
    smear with no disc structure at all.
    """
    ef = edit_frame
    n = len(arms)
    n_cols = n_cols or n
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell * n_cols + 0.6, cell * 0.82 * n_rows + 1.6),
        squeeze=False,
    )
    fig.patch.set_facecolor(BG)

    # Linear ramp: step 0 at 35% weight, the last step at 100%.
    w = np.linspace(0.35, 1.0, n_steps)
    w = w / w.sum()

    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols][k % n_cols]
        ax.set_facecolor(BG)
        if k >= n:
            ax.axis("off")
            continue
        arm = arms[k]
        src = (
            clean_obs[sample, ef : ef + n_steps]
            if arm.is_gt
            else arm.roll[sample, :n_steps]
        )
        comp = (src * w[:, None]).sum(0)
        # Renormalise to the peak so the trail uses the full display range; the
        # per-cell scale is therefore RELATIVE here, unlike frame_grid's fixed 0-1.
        comp = comp / max(comp.max(), 1e-6)
        ax.imshow(
            unflatten(comp, cfg),
            cmap=CMAP,
            vmin=0,
            vmax=1,
            origin="lower",
            extent=world_extent(cfg),
            aspect="equal",
            interpolation="nearest",
        )
        _locators(ax, target_xy, ghost_xy, cfg.radius)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2a2f45")
        lbl = arm.name + (f"\n{arm.metric}" if arm.metric else "")
        ax.set_title(
            lbl,
            color="w" if arm.is_gt else FG,
            fontsize=8.5,
            pad=4,
            fontweight="bold" if arm.is_gt else "normal",
        )

    _legend(fig)
    head = f"{fig_num} — {title}" if fig_num else title
    fig.suptitle(head, color="w", fontsize=11, y=0.995)
    fig.text(
        0.5,
        0.02,
        f"all {n_steps} rollout steps composited per arm; later steps brighter. "
        f"per-cell relative scale (unlike the fixed 0–1 of the frame grid).",
        color=FG,
        fontsize=7.5,
        ha="center",
    )
    # hspace is generous because the per-arm titles run to two or three lines and
    # collided with the row above them at the default spacing.
    fig.subplots_adjust(
        top=0.78, bottom=0.10, left=0.02, right=0.98, wspace=0.06, hspace=0.62
    )
    return fig


def frame_animation(
    arms: list[Arm],
    *,
    cfg: SimConfig,
    sample: int,
    ctx_obs: np.ndarray,  # (N, T, R) — the NOISY observations, edits.obs
    clean_obs: np.ndarray,  # (N, T, R) — clean render, for the GT arm only
    edit_frame: int,
    path: str,  # e.g. "anim3_editors.gif" — the number MUST match `anim_num`
    anim_num: str,  # e.g. "Anim 3"
    title: str = "",
    target_xy: np.ndarray | None = None,
    ghost_xy: np.ndarray | None = None,
    n_steps: int = 15,
    n_ctx: int = DEFAULT_CTX,
    fps: float = 3.0,
    hold_edit: int = 3,
    cell: float = 2.0,
    dpi: int = 110,
):
    """Animated GIF of the same comparison — arms side by side, time as the animation.

    The optional third view of the approved 2D spec (`WATERFALL_SPEC_2D.md`). It is an
    **addition to** `frame_grid` + `frame_trails`, never a substitute: a GIF cannot be
    read in a committed notebook diff or a paper, so a claim still ships with the pair.

    Where it earns its place is the thing a subsampled grid cannot show — the *motion*.
    Whether an edited object travels smoothly or teleports and snaps back is obvious in
    3 seconds of animation and genuinely hard to read off five stills.

    Obeys `CLAUDE.md`'s animation rules, which exist because the defaults get them wrong:

    * a **persistent figure-level title carrying the number** (`anim_num`), separate from
      the per-frame caption. `path`'s basename must carry the same number — asserted.
    * **~3 fps**, not matplotlib's default, so it is slow enough to read.
    * **holds on the key frames**: the last pre-edit frame and the edit frame (step 0) are
      each repeated `hold_edit` times, so the viewer can register the edit rather than
      having it flash past in one frame. **Do not "fix" the frame count** — the GIF
      encoder collapses each run of identical frames into one frame carrying the summed
      duration, so a 22-slot timeline is stored as 18 frames with two 990 ms holds. The
      pause is preserved exactly; only the encoding is compact (verified 2026-08-12).

    Content rules are identical to `frame_grid`: GT arm first showing `clean_obs`, the
    other arms their OWN free-run from step 0, **noisy** context frames before the edit,
    green target / red-dashed ghost circles, figure-top legend, fixed `vmin=0, vmax=1`,
    and `aspect="equal"` so the circles stay round.

    Returns the saved path.
    """
    import re

    from matplotlib.animation import FuncAnimation, PillowWriter

    num = re.search(r"\d+", anim_num)
    if num and num.group() not in str(path):
        raise ValueError(
            f"animation number and filename must match: {anim_num!r} vs {path!r} "
            "(CLAUDE.md — the saved file is named to match the figure number)"
        )
    if hold_edit < 1:
        raise ValueError(
            "hold_edit must be >= 1; the whole point is to pause on the edit"
        )

    ef = edit_frame
    # Build the timeline explicitly so the holds are auditable: context frames (with the
    # LAST one held), then the edit frame (held), then the remaining free-run steps.
    # `kind` is "ctx" or "run"; `k` indexes obs frames or rollout steps accordingly.
    timeline: list[tuple[str, int]] = []
    for j, t in enumerate(range(ef - n_ctx, ef)):
        reps = hold_edit if j == n_ctx - 1 else 1  # hold the last frame before the edit
        timeline += [("ctx", t)] * reps
    timeline += [("run", 0)] * hold_edit  # hold the edit frame itself
    timeline += [("run", s) for s in range(1, n_steps)]

    n = len(arms)
    fig, axes = plt.subplots(
        1, n, figsize=(cell * n + 0.4, cell * CELL_ASPECT + 1.35), squeeze=False
    )
    axes = axes[0]
    fig.patch.set_facecolor(BG)

    ims = []
    for ax, arm in zip(axes, arms):
        ax.set_facecolor(BG)
        im = ax.imshow(
            np.zeros(grid_shape(cfg)),
            cmap=CMAP,
            vmin=VMIN,
            vmax=VMAX,
            origin="lower",
            extent=world_extent(cfg),
            aspect="equal",
            interpolation="nearest",
        )
        ims.append(im)
        _locators(ax, target_xy, ghost_xy, cfg.radius)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2a2f45")
        lbl = arm.name + (f"\n{arm.metric}" if arm.metric else "")
        ax.set_title(
            lbl,
            color="w" if arm.is_gt else FG,
            fontsize=8,
            pad=4,
            fontweight="bold" if arm.is_gt else "normal",
        )

    # Title ABOVE the legend, matching `frame_grid` / `frame_trails` — the reading order
    # is what-is-shown, then how-to-read-it.
    # PERSISTENT figure-level title carrying the number — never per-frame, so the number
    # is legible in every frame of the GIF.
    fig.suptitle(f"{anim_num} — {title}", color="w", fontsize=10.5, y=0.975)
    _legend(fig, y=0.90)
    caption = fig.text(0.5, 0.035, "", color=FG, fontsize=8.5, ha="center")
    fig.subplots_adjust(top=0.66, bottom=0.11, left=0.02, right=0.98, wspace=0.06)

    def update(i):
        kind, k = timeline[i]
        for im, arm in zip(ims, arms):
            if kind == "ctx":
                # Every arm was teacher-forced on the same NOISY frames. Only the GT arm
                # is ever shown clean, and only after the edit.
                im.set_data(unflatten(ctx_obs[sample, k], cfg))
            elif arm.is_gt:
                im.set_data(unflatten(clean_obs[sample, ef + k], cfg))
            else:
                im.set_data(unflatten(arm.roll[sample, k], cfg))
        caption.set_text(
            f"t = {k if kind == 'ctx' else ef + k}"
            + (
                "   ·   pre-edit context (teacher-forced, noisy)"
                if kind == "ctx"
                else f"   ·   free-run step {k}"
            )
            + ("   ·   EDIT FRAME" if kind == "run" and k == 0 else "")
        )
        return [*ims, caption]

    anim = FuncAnimation(fig, update, frames=len(timeline), blit=False)
    anim.save(
        str(path),
        writer=PillowWriter(fps=fps),
        dpi=dpi,
        savefig_kwargs={"facecolor": BG},
    )
    plt.close(fig)
    return path
