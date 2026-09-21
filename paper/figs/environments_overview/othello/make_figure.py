"""Othello and its three rule variants (paper ``fig:othello_and_variants``): before / after boards under each of
the four rule sets, one move per panel, the discs it flips ringed.

Panels (a) standard, (b) adjacent-flip and (c) adjacent-noflip share ONE board: a position from a real standard
bench game (``load_benchmark("oth-uniform")``) that is provably reachable under the other two rule sets, so the
panels differ only in the legal squares and in what a move flips. Reachability is decided with the vendor's own
``OthelloBoardState``: ``reach`` is a depth-first search over placement orders from the opening that only lands
on the target's occupied squares (moves from ``get_valid_moves``, applied by ``umpire``) and keeps a child only
while every disc on the board has the target's colour, so under a flipping rule set only flip-free orders count
(a sufficient condition; an adjacency-legal standard prefix is accepted too, since the same sequence rebuilds
the same board under adjacent-flip). Visited states are skipped, so the search is exhaustive at these sizes
(at most ~1600 states). ``shared_board`` scans the positions after 14, 13, ... 4 moves of all 1000 bench games
and stops at the first move count with a survivor; among survivors it takes the one whose best shared move
(legal under all three rule sets) flips the most discs, then the largest legal-set difference, then the lowest
case id. Result (2026-09-21): nothing survives after 7 to 14 moves; three positions after 6 moves do, and case
182 is drawn. The placement orders are the evidence, recorded in ``boards.json``.

Panel moves: (a) the enclosure-legal move that flips the most discs; (b) the adjacency-legal move that flips the
most discs; (c) in ``composite_O2`` the same move as (b), so the only difference between (b) and (c) is that
nothing flips, and in ``composite_O2_altmove`` the move legal under all three rule sets that lies farthest from
(b)'s (it would flip a different disc under the flip rules, and flips nothing here). Panel (d) standard-noflip
uses its own bench game: the seeded rule (``--seed``, ``--move`` K, ``--min-flips``) is applied to every instance
in order and only standard-noflip is drawn.

Outputs: ``composite_O2`` and ``composite_O2_altmove`` (5.5 in wide, bold panel letters, key strip beneath),
``legend_key``, the before / after / pair boards under ``pieces/`` (2.0 in boards), ``boards.json``. Colours and
geometry come from ``paper_style`` and the qualitative Othello figure's ``draw_board``.

    .pim/bin/python paper/figs/environments_overview/othello/make_figure.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
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
from pim.environments.othello.data import canonical_vocab  # noqa: E402
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
STD, SHARED = "oth-uniform", ["oth-adjacent", "oth-adjacent-flip"]   # the rule sets that must reach the shared board
LEGAL_C, PLACED_C, FLIP_C = ps.BOARD_TINT, ps.ORIGIN_C, ps.DEST_C    # legal square, the move, a disc it flips
DOT_R, RING_R, GHOST_A = 0.11, DISC_R + 0.07, 0.5
BOARD_IN = 2.0                                           # a piece's board, inches on the page


# ── data: bench games replayed under an instance's rules ─────────────────────────────────
def histories(inst: str) -> list[list[int]]:
    itos = {v: k for k, v in canonical_vocab().items()}
    bench = load_benchmark(inst)
    hist = [None] * bench.n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    return hist


def position(inst: str, moves: list[int]) -> dict:
    """The board after ``moves`` under inst's rules, and what the last move did to it."""
    b = OthelloBoardState(**rules_of(inst))
    b.update(moves[:-1])
    regular = b.tentative_move(moves[-1]) == 1            # 2 would mean the mover had to pass
    before = b.state.copy()
    b.umpire(moves[-1])
    flipped = [sq for sq in range(64) if before.flat[sq] != 0 and b.state.flat[sq] != before.flat[sq]]
    return {"board": (b.state + 1).flatten().astype(int),  # white 0 / blank 1 / black 2, as draw_board reads it
            "placed": int(moves[-1]), "flipped": flipped, "legal": [int(m) for m in b.get_valid_moves()],
            "regular": regular, "n": len(moves), "mover": "black" if b.next_hand_color > 0 else "white"}


def pick(inst: str, hist: list[list[int]], k: int, rng: np.random.Generator, min_flips: int) -> int:
    """The own-game rule: the first seeded case whose moves K+1 and K+2 are regular and, where the rule set
    flips, whose move K+1 flips at least ``min_flips`` discs."""
    need = min_flips if rules_of(inst)["flip"] else 0
    for i in rng.permutation(len(hist)):
        h = hist[int(i)]
        Q, R = position(inst, h[:k + 1]), position(inst, h[:k + 2])
        if Q["regular"] and R["regular"] and len(Q["flipped"]) >= need:
            return int(i)
    raise RuntimeError(f"no case of {inst} satisfies the rule at move {k}")


# ── the shared board: a standard position provably reachable under the other rule sets ───
def match(b: OthelloBoardState, T: np.ndarray) -> bool:
    occ = b.state != 0
    return bool((b.state[occ] == T[occ]).all())


def reach(T: np.ndarray, rules: dict, cap: int = 20000) -> list[int] | None:
    """A placement order that rebuilds the target board T (8x8, +1 black / -1 white / 0 empty) under ``rules``
    by the generator's own game process, or None (see the module docstring)."""
    root = OthelloBoardState(**rules)
    if not match(root, T):
        return None
    n_target, stack, seen = int((T != 0).sum()), [(root, [])], set()
    while stack and len(seen) < cap:
        b, order = stack.pop()
        if int((b.state != 0).sum()) == n_target:
            return order
        for m in b.get_valid_moves():
            if T.flat[m] == 0:
                continue
            c = deepcopy(b)
            c.umpire(m)
            key = (c.state.tobytes(), c.next_hand_color)
            if match(c, T) and key not in seen:
                seen.add(key)
                stack.append((c, order + [m]))
    return None


def legal_sequence(moves: list[int], rules: dict) -> bool:
    """True if ``moves`` is a regular (no-pass) game under ``rules``: the same sequence then rebuilds the same board."""
    b = OthelloBoardState(**rules)
    for m in moves:
        if b.tentative_move(m) != 1:
            return False
        b.umpire(m)
    return True


def shared_board(hist: list[list[int]], ms, insts: list[str]) -> dict:
    """The standard bench position reachable under every rule set in ``insts`` (adjacency legality read off
    ``insts[0]``), chosen as the module docstring says. Returns the case, move count, the standard moves, the
    orders per instance, both legal sets, the best shared move and the survivors per move count scanned."""
    scan = {}
    for m in ms:
        found = []
        for i, h in enumerate(hist):
            b = OthelloBoardState(**rules_of(STD))
            b.update(h[:m])
            T, orders = b.state.copy(), {}
            for inst in insts:
                r = rules_of(inst)
                o = reach(T, r) or (h[:m] if r["flip"] and legal_sequence(h[:m], r) else None)
                if o is None:
                    break
                orders[inst] = o
            if len(orders) < len(insts):
                continue
            ba = OthelloBoardState(**rules_of(insts[0]))
            ba.update(orders[insts[0]])
            assert ba.next_hand_color == b.next_hand_color, "the same board must have the same mover"
            L_std, L_adj = set(b.get_valid_moves()), set(ba.get_valid_moves())
            both = sorted(L_std & L_adj)
            scan.setdefault(m, []).append(i)
            if not both:
                continue
            flips = {mv: len(position(STD, h[:m] + [mv])["flipped"]) for mv in both}
            mv = max(both, key=lambda s: (flips[s], -s))
            found.append(((flips[mv], len(L_std ^ L_adj), -i),
                          dict(case=i, m=m, moves=h[:m], orders=orders, legal_std=sorted(L_std), legal_adj=sorted(L_adj),
                               chosen=mv, survivors=scan)))
        if found:
            return max(found, key=lambda x: x[0])[1]
    raise RuntimeError(f"no shared board for {insts} at moves {list(ms)}")


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


def pair(P: dict, Q: dict) -> dict:
    """The O2 boards: before = the position with legal dots and the move as a ghost disc; after = the position
    after the move, the placed disc and the flipped discs ringed."""
    return {"before": dict(X=P, dots=True, ghost=(Q["placed"], Q["board"][Q["placed"]]), placed=Q["placed"]),
            "after": dict(X=Q, placed=Q["placed"], flipped=Q["flipped"])}


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


KEY_W, KEY_H = 3.4, 0.3          # the legend key strip, inches


def composite(all_views: dict, keys: list[str], stem: Path, *, arrows: bool, labels: list[str] | None = None,
              key: bool = True) -> None:
    """The four rule sets across (bold panel letters only), one row per key (a short label each if ``labels``),
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
        fig.text(x0 / W, 1 - 0.5 * top / H, f"({'abcd'[c]})", ha="left", va="center", fontsize=9, fontweight="bold")
        for r, k in enumerate(keys):
            y0 = rows_y[r]
            board(fig.add_axes([x0 / W, y0 / H, s / W, s / H]), **all_views[name][k], lw=1.1, dot_r=0.1)
            if arrows and r:
                arrow(fig, ((x0 + s / 2) / W, (y0 + s + rg - 0.05) / H), ((x0 + s / 2) / W, (y0 + s + 0.05) / H))
    for y0, label in zip(rows_y, labels or []):
        fig.text((left - 0.08) / W, (y0 + s / 2) / H, label, ha="right", va="center", fontsize=8)
    ps.save(fig, stem)
    plt.close(fig)


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
    ap.add_argument("--move", type=int, default=14, help="own game (d): the board after move K; the move shown is K+1")
    ap.add_argument("--min-flips", type=int, default=1, help="own game: discs move K+1 must flip, where the rule set flips")
    a = ap.parse_args()
    k, pieces = a.move, HERE / "pieces"
    pieces.mkdir(exist_ok=True)
    hist = {inst: histories(inst) for _, inst in VARIANTS}
    sq = lambda s: {"square": int(s), "name": permit_reverse(int(s))}          # noqa: E731
    names = lambda seq: [permit_reverse(int(s)) for s in seq]                 # noqa: E731

    def o2_pieces(v: dict, name: str, suffix: str = "") -> None:
        for key in ("before", "after"):
            row([v[key]], pieces / f"{name}_{key}{suffix}", arrows=False)
        row([v["before"], v["after"]], pieces / f"{name}_O2_pair{suffix}")

    # the shared board and the panel moves
    S = shared_board(hist[STD], range(14, 3, -1), SHARED)
    seqs = {STD: S["moves"], **S["orders"]}
    L_std, L_adj = set(S["legal_std"]), set(S["legal_adj"])
    n_flips = lambda mv: len(position(STD, S["moves"] + [mv])["flipped"])       # noqa: E731
    most = lambda L: max(L, key=lambda s: (n_flips(s), -s))                    # noqa: E731  the move flipping the most discs
    far = lambda s, t: max(abs(s // 8 - t // 8), abs(s % 8 - t % 8))            # noqa: E731  squares apart (king moves)
    mv_b = most(L_std & L_adj)      # (b): adjacency-only squares enclose nothing, so the most-flipping adjacency move is shared
    COMPOSITES = {"composite_O2": {STD: most(L_std), "oth-adjacent-flip": mv_b, "oth-adjacent": mv_b},
                  "composite_O2_altmove": {STD: most(L_std), "oth-adjacent-flip": mv_b,
                                           "oth-adjacent": max(L_std & L_adj, key=lambda s: (far(s, mv_b), -s))}}
    P_std = position(STD, S["moves"])
    V, panels = {c: {} for c in COMPOSITES}, {c: {} for c in COMPOSITES}
    for (name, inst), letter in zip(VARIANTS[:3], "abc"):
        P = position(inst, seqs[inst])
        assert (P["board"] == P_std["board"]).all() and P["mover"] == P_std["mover"], inst   # the evidence, re-checked
        for comp, mv in ((c, m[inst]) for c, m in COMPOSITES.items()):
            Q = position(inst, seqs[inst] + [mv])
            V[comp][name] = pair(P, Q)
            panels[comp][letter] = {"variant": name, "instance": inst, "board": "shared", "sequence": names(seqs[inst]),
                                    "legal_moves": names(P["legal"]), "move": sq(mv), "flipped": names(Q["flipped"])}
            print(f"{comp:<20} ({letter}) {name:<16} {len(P['legal']):2d} legal, move {permit_reverse(mv)} flips {names(Q['flipped'])}")
        o2_pieces(V["composite_O2"][name], name)
        if COMPOSITES["composite_O2_altmove"][inst] != COMPOSITES["composite_O2"][inst]:
            o2_pieces(V["composite_O2_altmove"][name], name, "_altmove")

    # panel (d): standard-noflip on its own game (the seeded rule applied to every instance in order)
    rng = np.random.default_rng(a.seed)
    own = {inst: pick(inst, hist[inst], k, rng, a.min_flips) for _, inst in VARIANTS}
    h = hist["oth-noflip"][own["oth-noflip"]]
    Pd, Qd = position("oth-noflip", h[:k]), position("oth-noflip", h[:k + 1])
    for comp in COMPOSITES:
        V[comp]["standard_noflip"] = pair(Pd, Qd)
        panels[comp]["d"] = {"variant": "standard_noflip", "instance": "oth-noflip", "board": "own game",
                             "bench_case_id": own["oth-noflip"], "board_after_move": k, "mover": Pd["mover"],
                             "legal_moves": names(Pd["legal"]), "move": {"number": k + 1, **sq(Qd["placed"])},
                             "flipped": names(Qd["flipped"])}
    o2_pieces(V["composite_O2"]["standard_noflip"], "standard_noflip")
    print(f"(d) standard_noflip own game case {own['oth-noflip']}, move {k + 1} = {permit_reverse(Qd['placed'])} by {Pd['mover']}, "
          f"{len(Pd['legal'])} legal; own-game picks in order {own}")

    legend_key(HERE / "legend_key")
    for comp in COMPOSITES:
        composite(V[comp], ["before", "after"], HERE / comp, arrows=True)
    json.dump({
        "seed": a.seed, "move": k, "min_flips": a.min_flips,
        "markers": {"legal move": LEGAL_C, "chosen move / placed disc": PLACED_C, "flipped disc": FLIP_C},
        "shared_board": {
            "bench_instance": STD, "bench_case_id": S["case"], "board_after_move": S["m"], "mover": P_std["mover"],
            "board_white0_blank1_black2": [int(x) for x in P_std["board"]], "standard_moves": names(S["moves"]),
            "placement_orders": {inst: names(o) for inst, o in S["orders"].items()},
            "adjacent_flip_evidence": ("the standard sequence itself is adjacency-legal"
                                       if S["orders"]["oth-adjacent-flip"] == S["moves"] else "a flip-free adjacency order"),
            "legal_standard": names(S["legal_std"]), "legal_adjacency": names(S["legal_adj"]),
            "legal_under_all_three": names(sorted(L_std & L_adj)),
            "search": {"moves_scanned": "14 down to 4, stopping at the first move count with a survivor",
                       "reachable_under_all_three_by_move": {str(m): ids for m, ids in S["survivors"].items()},
                       "condition": "exact for adjacent-noflip; for adjacent-flip a sufficient condition (flip-free "
                                    "order or adjacency-legal standard prefix)"}},
        "panel_moves": {"a": "the enclosure-legal move flipping the most discs", "b": "the adjacency-legal move flipping "
                        "the most discs", "c": {"composite_O2": "the same move as (b)", "composite_O2_altmove": "the move "
                        "legal under all three rule sets farthest from (b)'s"}, "d": "the own game's real move K+1"},
        "panels": panels,
        "own_game_rule": {"rule": "first case in a seeded permutation of the instance's 1000 bench cases whose moves "
                                  "K+1 and K+2 are regular (no pass) and, where the rule set flips, whose move K+1 flips "
                                  "at least min_flips discs; applied to every instance in VARIANTS order, only "
                                  "standard-noflip is drawn", "picks": own}},
        open(HERE / "boards.json", "w"), indent=1)
    print("->", HERE / "boards.json")
