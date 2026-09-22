"""Qualitative edits on the Othello variants — real mid-game boards, the model's next-move distribution.

One column (or row, ``--layout``) per variant. For each, ``--games`` board states drawn from the
canonical edit bench — real games, ``--min-moves``..``--max-moves`` in — chosen by ``--seed``, so the
variants do not share a scenario (their rules differ). Per board: Unedited (the pre-edit board, the
model's move distribution), Ground truth (the post-edit board, uniform over its legal moves — the
reference the Edit Index scores against), then PI / GS / IM (the post-edit board, the distribution
after each write at the run's scored best arm, symmetric-difference construction). Squares are tinted
by probability mass; the flipped tile and the squares whose legality it switches are outlined — cyan on
the pre-edit board, pink on the post-edit boards (``mark``, shared with the main-text figure).

Everything canonical comes from ``pim``: the bench and its legal sets (``load_benchmark``), boards
replayed by ``tokens_and_labels`` under the instance's rules, probes from the run's cache, the
writes from ``arms`` (``linear_arm`` pinv / ``grad_steer_arm`` / ``inverse_arms``), exactly as the
scorer calls them. Every write is computed once for all 1000 cases and cached in ``.scratch/``, so a
new seed or layout is a redraw. Output beside this script: ``othello_edits_seed<k>_<layout>.{pdf,png,json}``.

    python paper/figs/qualitative_edits_othello/make_figure.py --seed 0 --layout rows --games 2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # paper/figs
import paper_style as ps  # noqa: E402

ps.apply()                                  # one look for every paper figure (Arial, TrueType, white page)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

HERE = Path(__file__).resolve().parent
EI = "edit_index_symdiff"                   # the headline construction the arms are picked by
VARIANTS = [
    ("Standard", "initial_othello_comparison/L-oth-20m"),
    ("Adjacent Flip", "adjacent_flip_ablation/L-oth-adjacent-flip-20m"),
    ("Adjacent NoFlip", "adjacency_ablation/L-oth-adjacent-20m"),
    ("Standard NoFlip", "flip_ablation/L-oth-noflip-20m"),
]
CONDITIONS = ("Unedited", "Ground truth", "PI", "GS", "IM")
DEV = oa.DEV


# ── the two ground-truth conditions (Sevan, round 6, 2026-09-22) ─────────────────────────────────
# The second condition used to be the model's unedited prediction; it is now the TRUE pre-edit legal set, so the
# figure shows both reference worlds (the two the Edit Index compares against) and then the three edited writes.
UNEDITED_GT, EDITED_GT = "Unedited GT", "Edited GT"
DISPLAY = {"Unedited": UNEDITED_GT, "Ground truth": EDITED_GT}      # cache key -> drawn label
BOLD = ("Ground truth",)                                            # the Edited GT condition stays bold


def legal_uniform(legal, n_sq: int = 64) -> np.ndarray:
    """Uniform over one legal set: THE construction behind the Edited GT column (``compute`` builds
    ``probs["Ground truth"]`` from ``bench.legal_post`` with it), applied to either legal set."""
    p = np.zeros(n_sq, np.float32)
    if len(legal):
        p[list(legal)] = 1.0 / len(legal)
    return p


def cond_probs(col: dict, i: int, cond: str) -> np.ndarray:
    """The distribution drawn for one condition of one case. ``Unedited`` is the TRUE pre-edit legal set
    (uniform over ``legal_pre``, on the pre-edit board), not the model's unedited prediction, which the cache
    still carries under the same key for the sidecars' per-case index. Every other condition is the cached
    distribution: ``Ground truth`` the post-edit legal set, PI / GS / IM the model's after each write."""
    if cond == "Unedited":
        return legal_uniform(col["legal_pre"][i])
    return np.asarray(col["probs"][cond][i], float)


def best_arm(scores: dict, editor: str) -> dict:
    """The arm the TABLES report (``pim.metrics.selection.best_arm``: best Edit Index inside the fidelity
    guard, the unguarded best only if there is none). Until 2026-09-19: the unguarded argmax."""
    from pim.metrics.selection import best_arm as select

    return select(scores["arms"], editor, EI)


@torch.no_grad()
def compute(run: str) -> dict:
    """Every case of the run's bench: boards before / after the edit, legal sets, and the move
    distributions unedited and under PI / GS / IM at the scored best arms."""
    run_dir = REPO / "runs" / run
    scores = json.loads((run_dir / "scores.json").read_text())
    S = scores["settings"]
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    n_games = S["oth_probe_games"]
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n_games], ln[:n_games])], **rules)
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    n = bench.n_cases
    # the histories, replayed under the instance's rules → absolute boards (white 0 / blank 1 / black 2)
    hist = [None] * n
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels(hist, **rules)
    board_pre = np.stack([bd.labels[i, len(hist[i]) - 1] for i in range(n)])          # (n, 64)
    board_post = board_pre.copy()
    ar = np.arange(n)
    board_post[ar, bench.pos_int] = 2 - board_pre[ar, bench.pos_int]                   # the flip, absolute
    assert (board_pre[ar, bench.pos_int] != 1).all(), "an edited tile must be occupied"
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)      # cache hit
    NP = model.n_layers + 1
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}
    uns = oa.unsteered_probs(model, bench)
    arms = {ed: best_arm(scores, ed) for ed in ("PI", "GS", "IM")}
    pi, gs, im = arms["PI"], arms["GS"], arms["IM"]
    probs = {"Unedited": uns}
    probs["PI"] = oa.linear_arm(model, bench, lin, tgt, cur, mode="pinv", alpha=pi["alpha"], points={pi["point"]})[0]
    probs["GS"] = oa.grad_steer_arm(model, bench, mlp, gs["point"], alpha=gs["alpha"], n_steps=S["oth_gs_steps"],
                                    beta=S["oth_gs_beta"], target_labels=tgt)[0]
    _, _, pb = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=run_dir / "probes", n_games=n_games,
                               points=[im["point"]], uns_probs=uns, log=None, return_probs=True)
    probs["IM"] = pb[("IM", im["point"])]
    probs["Ground truth"] = np.stack([legal_uniform(lp) for lp in bench.legal_post])
    lengths = np.array([len(h) for h in hist])
    return {"run": run, "instance": inst, "board_pre": board_pre, "board_post": board_post, "pos": bench.pos_int.copy(),
            "legal_pre": [list(x) for x in bench.legal_pre], "legal_post": [list(x) for x in bench.legal_post],
            "lengths": lengths, "probs": probs, "arms": {e: (int(a["point"]), float(a["alpha"])) for e, a in arms.items()},
            "ei": {e: float(a[EI]) for e, a in arms.items()}}


# ── drawing ──────────────────────────────────────────────────────────────────────────
GREEN, LINE, TINT, EDIT_C = ps.BOARD_GREEN, ps.BOARD_LINE, ps.BOARD_TINT, ps.BOARD_EDIT
DISC_R = 0.38
TINT_SCALE = 0.02     # the probability at which a square is fully tinted (Sevan: 10x louder than 0.2)

# ── the marking (Sevan, 2026-09-21) — one helper for the appendix and the main-text figure ──────────
# On the PRE-edit board (Unedited) the flipped tile and every square whose legality the flip switches
# (legal_pre XOR legal_post — exactly the squares the symmetric-difference Edit Index scores) are outlined
# in cyan (``ps.ORIGIN_C``); on every POST-edit board the same squares in pink (``ps.DEST_C``). Nothing
# else is outlined; the yellow tint stays.
MARK_LW = 1.6


def marked_squares(pos: int, legal_pre, legal_post) -> list[int]:
    """The flipped tile plus the symmetric difference of the two legal sets."""
    return sorted({int(pos)} | (set(map(int, legal_pre)) ^ set(map(int, legal_post))))


def mark_color(cond: str) -> str:
    return ps.ORIGIN_C if cond == "Unedited" else ps.DEST_C


def mark(ax, squares, color: str, lw: float = MARK_LW) -> None:
    for sq in squares:
        r, c = divmod(int(sq), 8)
        ax.add_patch(Rectangle((c, 7 - r), 1, 1, facecolor="none", edgecolor=color, linewidth=lw, zorder=5))


def mark_key(owner, *, fontsize: float = 8, markersize: float = 5, **kw):
    """The key, once per figure: a cyan DOT "pre-edit", a pink DOT "post-edit" (Sevan, round 3; the "gt"
    suffix was dropped in round 7). ``owner`` is a Figure, SubFigure or Axes; ``kw`` goes to its ``legend``."""
    handles = [Line2D([], [], marker="o", linestyle="none", color=c, markersize=markersize) for c in (ps.ORIGIN_C, ps.DEST_C)]
    return owner.legend(handles=handles, labels=["pre-edit", "post-edit"], fontsize=fontsize, handlelength=0.8,
                        handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0, **kw)


def draw_board(ax, board: np.ndarray, probs: np.ndarray, edited: int | None, *, gamma: float = 0.6,
               tint_scale: float = TINT_SCALE, locator: bool = True):
    """One 8×8 board: green squares tinted by probability mass — fully at ``tint_scale`` and above,
    (p / tint_scale) ** gamma below — black / white discs, the edited tile outlined. ``board`` white 0 /
    blank 1 / black 2 (row-major, a1 top-left as in the data)."""
    g, t = np.array(to_rgb(GREEN)), np.array(to_rgb(TINT))
    for sq in range(64):
        r, c = divmod(sq, 8)
        a = min(1.0, float(probs[sq]) / tint_scale) ** gamma if probs[sq] > 0 else 0.0
        ax.add_patch(Rectangle((c, 7 - r), 1, 1, facecolor=(1 - a) * g + a * t, edgecolor=LINE, linewidth=0.4))
        if board[sq] != 1:
            ax.add_patch(Circle((c + 0.5, 7 - r + 0.5), DISC_R, facecolor="black" if board[sq] == 2 else "white",
                                edgecolor="#333333", linewidth=0.5, zorder=3))
    if locator and edited is not None:
        r, c = divmod(int(edited), 8)
        ax.add_patch(Rectangle((c, 7 - r), 1, 1, facecolor="none", edgecolor=EDIT_C, linewidth=2.2, zorder=5))
    ax.set_xlim(0, 8); ax.set_ylim(0, 8); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def draw(cols: dict, picks: dict, out: Path, *, layout: str = "rows", gamma: float = 0.6, tint_scale: float = TINT_SCALE,
         locator: bool = True,
         title_size: float = 13, label_size: float = 11):
    names = [v[0] for v in VARIANTS]
    n_games = len(next(iter(picks.values())))
    conds = list(CONDITIONS)
    if layout == "cols":                     # the sketch: variants across, conditions down
        ncol, nrow = len(names) * n_games, len(conds)
        cell = lambda v, k, c: (c, v * n_games + k)          # noqa: E731
        col_titles = [(v * n_games + k, names[v] if n_games == 1 else f"{names[v]}  ·  game {k + 1}")
                      for v in range(len(names)) for k in range(n_games)]
        row_labels = [(c, conds[c]) for c in range(len(conds))]
    else:                                    # transposed: conditions across (per game), variants down
        ncol, nrow = len(conds) * n_games, len(names)
        cell = lambda v, k, c: (v, k * len(conds) + c)       # noqa: E731
        col_titles = [(k * len(conds) + c, conds[c]) for k in range(n_games) for c in range(len(conds))]
        row_labels = [(v, names[v]) for v in range(len(names))]
    fig = plt.figure(figsize=(1.55 * ncol + 1.6, 1.55 * nrow + 0.9), facecolor="white")
    gs = GridSpec(nrow, ncol, figure=fig, left=0.11, right=0.995, top=0.92, bottom=0.01, wspace=0.08, hspace=0.08)
    axes = {}
    for v, name in enumerate(names):
        col = cols[name]
        for k, i in enumerate(picks[name]):
            for c, cond in enumerate(conds):
                r_, c_ = cell(v, k, c)
                ax = fig.add_subplot(gs[r_, c_])
                board = col["board_pre"] if cond == "Unedited" else col["board_post"]
                draw_board(ax, board[i], cond_probs(col, i, cond), col["pos"][i], gamma=gamma, tint_scale=tint_scale,
                           locator=False)
                if locator:
                    mark(ax, marked_squares(col["pos"][i], col["legal_pre"][i], col["legal_post"][i]), mark_color(cond),
                         lw=MARK_LW * 1.6)                        # the appendix page is drawn large and scaled down
                axes[(r_, c_)] = ax
    shown = lambda t: DISPLAY.get(t, t)   # noqa: E731
    for c_, text in col_titles:
        axes[(0, c_)].set_title(shown(text), fontsize=title_size, pad=6, color="black",
                                fontweight="bold" if text in BOLD else "normal")
    if layout == "rows" and n_games > 1:     # a game label above each group of five
        for k in range(n_games):
            a0, a1 = axes[(0, k * len(conds))], axes[(0, (k + 1) * len(conds) - 1)]
            x = (a0.get_position().x0 + a1.get_position().x1) / 2
            fig.text(x, 0.975, f"game {k + 1}", ha="center", va="center", fontsize=title_size, color="black")
    for r_, text in row_labels:
        ax = axes[(r_, 0)]
        y = (ax.get_position().y0 + ax.get_position().y1) / 2
        fig.text(ax.get_position().x0 - 0.008, y, shown(text), ha="right", va="center", fontsize=label_size, color="black",
                 fontweight="bold" if text in BOLD else "normal")
    if locator:                                   # the key, in the free corner above the row labels
        mark_key(fig, markersize=9, fontsize=label_size - 1, loc="upper left", bbox_to_anchor=(0.005, 0.995))
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0, facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=1, help="board states per variant")
    ap.add_argument("--layout", choices=("rows", "cols"), default="rows",
                    help="rows: variants down, conditions across (wide); cols: the sketch's arrangement")
    ap.add_argument("--min-moves", type=int, default=12)
    ap.add_argument("--max-moves", type=int, default=45)
    ap.add_argument("--gamma", type=float, default=0.6, help="tint = (p / tint-scale) ** gamma, capped at 1")
    ap.add_argument("--tint-scale", type=float, default=TINT_SCALE, help="probability at which a square is fully tinted")
    ap.add_argument("--no-locator", action="store_true", help="drop the cyan / pink marks (flipped tile + changed squares)")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached writes")
    ap.add_argument("--out-dir", default=None, help="write the outputs here instead of beside this script")
    a = ap.parse_args()
    cache = REPO / ".scratch" / "othello_edits_guarded_cache.pkl"      # writes at the guarded arms (2026-09-19)
    cols = pickle.load(open(cache, "rb")) if cache.exists() and not a.recompute else {}
    for name, run in VARIANTS:
        if name not in cols or cols[name]["run"] != run:
            print(f"computing {name} ({run}) …", flush=True)
            cols[name] = compute(run)
            cache.parent.mkdir(exist_ok=True)
            pickle.dump(cols, open(cache, "wb"))
            torch.cuda.empty_cache()
    rng = np.random.default_rng(a.seed)
    picks = {}
    for name, _ in VARIANTS:
        L = cols[name]["lengths"]
        ok = np.where((L >= a.min_moves) & (L <= a.max_moves))[0]
        picks[name] = [int(x) for x in rng.choice(ok, size=a.games, replace=False)]
        print(f"  {name:<16} cases {picks[name]}  moves in {[int(L[i]) for i in picks[name]]}  "
              f"arms {cols[name]['arms']}  EI {{{', '.join(f'{e} {x:+.2f}' for e, x in cols[name]['ei'].items())}}}")
    out_dir = Path(a.out_dir) if a.out_dir else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"othello_edits_seed{a.seed}_{a.layout}"
    draw(cols, picks, out, layout=a.layout, gamma=a.gamma, tint_scale=a.tint_scale, locator=not a.no_locator)
    json.dump({"seed": a.seed, "layout": a.layout, "picks": picks,
               "arms": {n: cols[n]["arms"] for n in picks}, "moves": {n: [int(cols[n]["lengths"][i]) for i in picks[n]] for n in picks}},
              open(out.with_suffix(".json"), "w"), indent=1)
    print("→", out.with_suffix(".pdf"), "and .png / .json")
