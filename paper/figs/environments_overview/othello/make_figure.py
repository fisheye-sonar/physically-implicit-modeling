"""Othello and its three rule variants (paper ``fig:othello_and_variants``): real mid-game boards from each
instance's edit bench, the mover's legal squares, one move and the discs it flips.

Per rule set (standard, adjacent-flip, adjacent-noflip, standard-noflip) one bench game is replayed under the
instance's own rules (``rules_of`` + the vendored ``OthelloBoardState``) to the board after move K
(``--move``); the game's real move K+1 is the chosen move. Three ways to convey the rule, all from the same
case: O1 one board (legal dots, the chosen move as a ghost disc, the discs it would flip ringed); O2 a
before / after pair; O3 three consecutive positions. Every board is its own PDF + PNG under ``pieces/``, plus
a legend key and one full-width composite per option; ``boards.json`` records instance, case id, move number,
chosen move and flipped squares for every piece.

Selection (``--seed``): the first case, in a seeded permutation of the 1000 bench cases, whose moves K+1 and
K+2 are regular moves (no pass) and, where the rule set flips, whose move K+1 flips at least ``--min-flips``
discs. Colours and geometry come from ``paper_style`` and the qualitative Othello figure's ``draw_board``.

    .pim/bin/python paper/figs/environments_overview/othello/make_figure.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path[:0] = [str(REPO / "paper" / "figs"), str(REPO)]
import paper_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

from pim.environments.othello.bench import load_benchmark  # noqa: E402
from pim.environments.othello.corpus import rules_of  # noqa: E402
from pim.environments.othello.data import canonical_vocab, synthetic_games  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState, permit_reverse  # noqa: E402


def _import(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_qual = _import(REPO / "paper" / "figs" / "qualitative_edits_othello" / "make_figure.py")
draw_board, DISC_R = _qual.draw_board, _qual.DISC_R      # the canonical board drawing
ps.apply()                                               # the reference module sets its own rcParams at import

VARIANTS = [("standard", "oth-uniform"), ("adjacent_flip", "oth-adjacent-flip"),
            ("adjacent_noflip", "oth-adjacent"), ("standard_noflip", "oth-noflip")]
LEGAL_C, PLACED_C, FLIP_C = ps.BOARD_TINT, ps.ORIGIN_C, ps.DEST_C    # legal square, the move, a disc it flips
DOT_R, RING_R, GHOST_A = 0.11, DISC_R + 0.07, 0.5
BOARD_IN = 2.0                                           # a piece's board, inches on the page


# ── data: the bench games, replayed under the instance's rules ───────────────────────────
def histories(inst: str) -> list[list[int]]:
    itos = {v: k for k, v in canonical_vocab().items()}
    bench = load_benchmark(inst)
    hist = [None] * bench.n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    return hist


def replay(inst: str, h: list[int], t: int) -> dict:
    """The board after the first t moves of h under inst's rules, and what move t did to it."""
    b = OthelloBoardState(**rules_of(inst))
    b.update(h[:t - 1])
    regular = b.tentative_move(h[t - 1]) == 1            # 2 would mean the mover had to pass
    before = b.state.copy()
    b.umpire(h[t - 1])
    flipped = [sq for sq in range(64) if before.flat[sq] != 0 and b.state.flat[sq] != before.flat[sq]]
    return {"board": (b.state + 1).flatten().astype(int),  # white 0 / blank 1 / black 2, as draw_board reads it
            "placed": int(h[t - 1]), "flipped": flipped, "legal": [int(m) for m in b.get_valid_moves()],
            "regular": regular, "t": t}


def pick(inst: str, hist: list[list[int]], k: int, rng: np.random.Generator, min_flips: int) -> int:
    need = min_flips if rules_of(inst)["flip"] else 0
    for i in rng.permutation(len(hist)):
        h = hist[int(i)]
        Q, R = replay(inst, h, k + 1), replay(inst, h, k + 2)
        if Q["regular"] and R["regular"] and len(Q["flipped"]) >= need:
            return int(i)
    raise RuntimeError(f"no case of {inst} satisfies the rule at move {k}")


def views(P: dict, Q: dict, R: dict) -> dict:
    """What each board shows. P / Q / R: the positions after moves K, K+1, K+2."""
    ghost = (Q["placed"], Q["board"][Q["placed"]])
    return {"legal": dict(X=P, dots=True),
            "O1": dict(X=P, dots=True, ghost=ghost, placed=Q["placed"], flipped=Q["flipped"]),
            "before": dict(X=P, dots=True, ghost=ghost, placed=Q["placed"]),
            "after": dict(X=Q, placed=Q["placed"], flipped=Q["flipped"]),
            "O3_1": dict(X=P, dots=True, placed=P["placed"], flipped=P["flipped"]),
            "O3_2": dict(X=Q, dots=True, placed=Q["placed"], flipped=Q["flipped"]),
            "O3_3": dict(X=R, dots=True, placed=R["placed"], flipped=R["flipped"])}


# ── drawing ──────────────────────────────────────────────────────────────────────────────
def xy(sq: int) -> tuple[float, float]:
    r, c = divmod(int(sq), 8)
    return c + 0.5, 7.5 - r


def board(ax, X: dict, *, probs=None, dots: bool = False, ghost=None, placed=None, flipped=(), lw: float = 1.4,
          dot_r: float = DOT_R) -> None:
    """draw_board's board (squares tinted by ``probs`` if given, as the qualitative figure tints them), then
    the rule markers: a dot on each legal square, a half-transparent disc where the chosen move would land,
    a ring on the placed disc and on every disc the move flips."""
    draw_board(ax, X["board"], np.zeros(64) if probs is None else probs, None)
    for sq in X["legal"] if dots else ():
        if ghost is None or sq != ghost[0]:              # the ghost disc stands in for its own dot
            ax.add_patch(Circle(xy(sq), dot_r, facecolor=LEGAL_C, edgecolor="none", zorder=4))
    if ghost is not None:
        sq, colour = ghost
        ax.add_patch(Circle(xy(sq), DISC_R, facecolor="black" if colour == 2 else "white", alpha=GHOST_A,
                            edgecolor="#333333", linewidth=0.5, zorder=3))
    rings = ([(placed, PLACED_C)] if placed is not None else []) + [(sq, FLIP_C) for sq in flipped]
    for sq, colour in rings:
        ax.add_patch(Circle(xy(sq), RING_R, facecolor="none", edgecolor=colour, linewidth=lw, zorder=6))


def arrow(fig, p0, p1) -> None:
    fig.add_artist(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9,
                                   color=ps.TEXT, linewidth=0.8, shrinkA=0, shrinkB=0))


def row(specs: list[dict], stem: Path, *, size: float = BOARD_IN, gap: float = 0.45, arrows: bool = True) -> None:
    """One or more boards side by side, an arrow between consecutive boards."""
    n = len(specs)
    W = n * size + (n - 1) * gap
    fig = plt.figure(figsize=(W, size))
    for j, spec in enumerate(specs):
        x0 = j * (size + gap)
        board(fig.add_axes([x0 / W, 0, size / W, 1]), **spec)
        if arrows and j:
            arrow(fig, ((x0 - gap + 0.08) / W, 0.5), ((x0 - 0.08) / W, 0.5))
    ps.save(fig, stem)
    plt.close(fig)


def composite(all_views: dict, keys: list[str], stem: Path, *, arrows: bool, labels: list[str] | None = None,
              key: bool = True) -> None:
    """The four rule sets across (panel letters only), one row per key (a short label each if ``labels``),
    the legend key strip beneath if ``key``, at the text width."""
    W, cg, top = ps.TEXT_WIDTH_IN, 0.12, 0.2
    left = 0.62 if labels else 0.0
    rg = 0.34 if arrows else 0.12
    s = (W - left - 3 * cg) / 4
    n = len(keys)
    bottom = KEY_H + 0.1 if key else 0.0
    H = top + n * s + (n - 1) * rg + bottom
    fig = plt.figure(figsize=(W, H))
    if key:
        key_entries(fig.add_axes([(W - KEY_W) / 2 / W, 0, KEY_W / W, KEY_H / H]))
    rows_y = [H - top - (r + 1) * s - r * rg for r in range(n)]
    for c, (name, _) in enumerate(VARIANTS):
        x0 = left + c * (s + cg)
        fig.text(x0 / W, 1 - 0.5 * top / H, f"({'abcd'[c]})", ha="left", va="center", fontsize=9)
        for r, key in enumerate(keys):
            y0 = rows_y[r]
            board(fig.add_axes([x0 / W, y0 / H, s / W, s / H]), **all_views[name][key], lw=1.1, dot_r=0.1)
            if arrows and r:
                arrow(fig, ((x0 + s / 2) / W, (y0 + s + rg - 0.05) / H), ((x0 + s / 2) / W, (y0 + s + 0.05) / H))
    for y0, label in zip(rows_y, labels or []):
        fig.text((left - 0.08) / W, (y0 + s / 2) / H, label, ha="right", va="center", fontsize=8)
    ps.save(fig, stem)
    plt.close(fig)


KEY_W, KEY_H = 3.4, 0.3          # the legend key strip, inches


def key_entries(ax) -> None:
    """Three entries, each marker on its own board tile: legal move, chosen move, flipped disc. ``ax`` spans
    KEY_W x KEY_H inches; data units are tile heights."""
    ax.set_xlim(0, KEY_W / KEY_H), ax.set_ylim(0, 1), ax.set_axis_off()
    t = 0.8
    for x, label in zip((0.15, 3.75, 7.55), ("legal move", "chosen move", "flipped disc")):
        cx, cy = x + t / 2, 0.5
        ax.add_patch(Rectangle((x, cy - t / 2), t, t, facecolor=ps.BOARD_GREEN, edgecolor=ps.BOARD_LINE, linewidth=0.4))
        if label == "legal move":
            ax.add_patch(Circle((cx, cy), DOT_R * t, facecolor=LEGAL_C, edgecolor="none"))
        else:
            ax.add_patch(Circle((cx, cy), DISC_R * t, facecolor="black", edgecolor="#333333", linewidth=0.5,
                                alpha=GHOST_A if label == "chosen move" else 1.0))
            ax.add_patch(Circle((cx, cy), RING_R * t, facecolor="none", linewidth=1.4,
                                edgecolor=PLACED_C if label == "chosen move" else FLIP_C))
        ax.text(x + t + 0.25, cy, label, ha="left", va="center", fontsize=8)


def legend_key(stem: Path) -> None:
    fig = plt.figure(figsize=(KEY_W, KEY_H))
    key_entries(fig.add_axes([0, 0, 1, 1]))
    ps.save(fig, stem)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--move", type=int, default=14, help="K: the board after move K; the chosen move is move K+1")
    ap.add_argument("--min-flips", type=int, default=1, help="discs move K+1 must flip, where the rule set flips")
    a = ap.parse_args()
    k = a.move
    pieces = HERE / "pieces"
    pieces.mkdir(exist_ok=True)
    rng = np.random.default_rng(a.seed)
    all_views, record = {}, {}
    for name, inst in VARIANTS:
        hist = histories(inst)
        i = pick(inst, hist, k, rng, a.min_flips)
        h = hist[i]
        P, Q, R = (replay(inst, h, t) for t in (k, k + 1, k + 2))
        all_views[name] = V = views(P, Q, R)
        sq = lambda s: {"square": int(s), "name": permit_reverse(int(s))}          # noqa: E731
        record[name] = {
            "instance": inst, "rules": rules_of(inst), "bench_case_id": i, "board_after_move": k,
            "mover": "black" if Q["board"][Q["placed"]] == 2 else "white",
            "legal_moves": [sq(s) for s in P["legal"]],
            "chosen_move": {"number": k + 1, **sq(Q["placed"])}, "flipped_by_chosen_move": [sq(s) for s in Q["flipped"]],
            "frames": {f"O3_{j + 1}": {"after_move": X["t"], "placed": sq(X["placed"]), "flipped": [sq(s) for s in X["flipped"]],
                                      "n_legal_next": len(X["legal"])} for j, X in enumerate((P, Q, R))},
            "pieces": {f"{name}_{key}": f"pieces/{name}_{key}.pdf" for key in list(V) + ["O2_pair", "O3_seq"]}}
        print(f"{name:<16} {inst:<18} case {i:4d}  move {k + 1} = {permit_reverse(Q['placed'])} by "
              f"{record[name]['mover']}, flips {[permit_reverse(s) for s in Q['flipped']]}, "
              f"{len(P['legal'])} legal moves; frame flips {[len(X['flipped']) for X in (P, Q, R)]}")
        for key, spec in V.items():
            row([spec], pieces / f"{name}_{key}", arrows=False)
        row([V["before"], V["after"]], pieces / f"{name}_O2_pair")
        row([V["O3_1"], V["O3_2"], V["O3_3"]], pieces / f"{name}_O3_seq", gap=0.35)
        # extra: the final position of one whole game from the instance's generator (standard-noflip's is
        # the checkerboard every game ends in)
        g = synthetic_games(1, seed=a.seed, n_workers=1, **rules_of(inst))[0]
        row([dict(X=replay(inst, g, len(g)))], pieces / f"{name}_terminal", arrows=False)
        record[name]["terminal"] = {"source": f"synthetic_games(1, seed={a.seed}, **rules_of(inst))[0]",
                                    "n_moves": len(g), "piece": f"pieces/{name}_terminal.pdf"}
    legend_key(HERE / "legend_key")
    composite(all_views, ["O1"], HERE / "composite_O1", arrows=False)
    composite(all_views, ["before", "after"], HERE / "composite_O2", arrows=True)
    composite(all_views, ["O3_1", "O3_2", "O3_3"], HERE / "composite_O3", arrows=True)
    json.dump({"seed": a.seed, "move": k, "min_flips": a.min_flips,
               "selection_rule": "first case in a seeded permutation of the instance's 1000 bench cases whose moves "
                                 "K+1 and K+2 are regular (no pass) and, where the rule set flips, whose move K+1 "
                                 "flips at least min_flips discs; the chosen move is the game's real move K+1",
               "markers": {"legal move": LEGAL_C, "chosen move / placed disc": PLACED_C, "flipped disc": FLIP_C},
               "variants": record}, open(HERE / "boards.json", "w"), indent=1)
    print("->", HERE / "boards.json")
