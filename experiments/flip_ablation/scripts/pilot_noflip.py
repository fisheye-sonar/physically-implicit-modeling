#!/usr/bin/env python
"""Pilot: what do no-flip Othello games look like? (before generating 20M of them)

Same generator, same seeds, both rule sets on N games: length distribution, passes per
game (moves where the mover was NOT the alternating player), the legal-set size along the
game and its Bayes floors (E[log|legal|] = the CE floor, E[1/|legal|] = the top-1 floor),
board fill and colour balance at the end, distinct games, and how often mine/theirs equals
the pure parity rule "the disc's colour is the parity of the offset at which its square was
played" (what a right-aligned linear observation probe can read).

Output: experiments/flip_ablation/scores/pilot_noflip.json + a printed comparison.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello.data import synthetic_games  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

EXP = REPO / "experiments" / "flip_ablation"


def analyse(games, flip: bool) -> dict:
    lens = np.array([len(g) for g in games])
    passes, legal_sizes, log_legal, inv_legal = [], [], [], []
    fill, black_share, parity_ok, parity_n = [], [], 0, 0
    by_t = {}
    for g in games:
        b = OthelloBoardState(flip=flip)
        expected = 1
        n_pass = 0
        placed_by = {}                                  # square -> move index
        for t, mv in enumerate(g):
            legal = b.get_valid_moves()
            L = len(legal)
            legal_sizes.append(L)
            log_legal.append(np.log(L))
            inv_legal.append(1.0 / L)
            by_t.setdefault(t, []).append(L)
            mover = b.next_hand_color if b.tentative_move(mv) == 1 else -b.next_hand_color
            if mover != expected:
                n_pass += 1
                expected = mover
            b.umpire(mv)
            placed_by[mv] = t
            expected = -mover
            # parity rule for the mine/theirs frame at this position: a square played at
            # move s is "mine" (belongs to the player to move next) iff (t - s) is odd
            nxt = b.next_hand_color
            for sq, s in placed_by.items():
                is_mine = b.state[sq // 8, sq % 8] == nxt
                parity_ok += int(is_mine == (((t - s) % 2) == 1))
                parity_n += 1
        passes.append(n_pass)
        fill.append(int((b.state != 0).sum()))
        black_share.append(float((b.state == 1).sum() / max((b.state != 0).sum(), 1)))
    hist = Counter(lens.tolist())
    return {
        "flip": flip, "n_games": int(len(games)),
        "length": {"mean": float(lens.mean()), "min": int(lens.min()), "p05": float(np.percentile(lens, 5)),
                   "p50": float(np.median(lens)), "p95": float(np.percentile(lens, 95)), "max": int(lens.max()),
                   "share_60": float((lens == 60).mean()), "histogram": {int(k): int(v) for k, v in sorted(hist.items())}},
        "passes_per_game": {"mean": float(np.mean(passes)), "share_zero": float(np.mean(np.array(passes) == 0)),
                            "max": int(max(passes))},
        "legal_set_size": {"mean": float(np.mean(legal_sizes)), "min": int(min(legal_sizes)),
                           "by_move": {int(t): float(np.mean(v)) for t, v in sorted(by_t.items())}},
        "bayes_ce": float(np.mean(log_legal)), "bayes_top1": float(np.mean(inv_legal)),
        "final_fill": {"mean": float(np.mean(fill)), "min": int(min(fill))},
        "black_share_final": float(np.mean(black_share)),
        "distinct_games": int(len({tuple(g) for g in games})),
        "mine_theirs_equals_parity_rule": float(parity_ok / max(parity_n, 1)),
        "tokens_total": int(lens.sum()),
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    t0 = time.time()
    res = {}
    for flip in (True, False):
        games = synthetic_games(n, seed=0, flip=flip)
        res["flip" if flip else "noflip"] = analyse(games, flip)
        print(f"  {'flip' if flip else 'noflip'}: {n} games analysed [{(time.time() - t0) / 60:.1f} min]", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores").mkdir(exist_ok=True)
    (EXP / "scores" / "pilot_noflip.json").write_text(json.dumps(res, indent=1))
    F, N = res["flip"], res["noflip"]
    rows = [("games", F["n_games"], N["n_games"]),
            ("length mean / p05 / p50 / p95", f"{F['length']['mean']:.1f} / {F['length']['p05']:.0f} / {F['length']['p50']:.0f} / {F['length']['p95']:.0f}",
             f"{N['length']['mean']:.1f} / {N['length']['p05']:.0f} / {N['length']['p50']:.0f} / {N['length']['p95']:.0f}"),
            ("length min / max", f"{F['length']['min']} / {F['length']['max']}", f"{N['length']['min']} / {N['length']['max']}"),
            ("share of games reaching 60 moves", f"{F['length']['share_60']:.3f}", f"{N['length']['share_60']:.3f}"),
            ("passes per game (mean / share with none / max)", f"{F['passes_per_game']['mean']:.3f} / {F['passes_per_game']['share_zero']:.3f} / {F['passes_per_game']['max']}",
             f"{N['passes_per_game']['mean']:.3f} / {N['passes_per_game']['share_zero']:.3f} / {N['passes_per_game']['max']}"),
            ("legal-set size, mean", f"{F['legal_set_size']['mean']:.2f}", f"{N['legal_set_size']['mean']:.2f}"),
            ("Bayes CE floor  E[log|legal|]", f"{F['bayes_ce']:.4f}", f"{N['bayes_ce']:.4f}"),
            ("Bayes top-1 floor  E[1/|legal|]", f"{F['bayes_top1']:.4f}", f"{N['bayes_top1']:.4f}"),
            ("final board fill (mean / min of 64)", f"{F['final_fill']['mean']:.1f} / {F['final_fill']['min']}", f"{N['final_fill']['mean']:.1f} / {N['final_fill']['min']}"),
            ("black share of discs at the end", f"{F['black_share_final']:.3f}", f"{N['black_share_final']:.3f}"),
            ("distinct games", F["distinct_games"], N["distinct_games"]),
            ("mine/theirs == parity-of-offset rule", f"{F['mine_theirs_equals_parity_rule']:.4f}", f"{N['mine_theirs_equals_parity_rule']:.4f}"),
            ("tokens total", F["tokens_total"], N["tokens_total"])]
    print(f"\n{'':48s} {'flip (oth-uniform)':>22s} {'noflip':>22s}")
    for name, a, b in rows:
        print(f"{name:48s} {str(a):>22s} {str(b):>22s}")
    print("\nlegal-set size by move (mean), noflip vs flip, every 5th move:")
    for t in range(0, 60, 5):
        f = F["legal_set_size"]["by_move"].get(t)
        g = N["legal_set_size"]["by_move"].get(t)
        print(f"  move {t:>2}: flip {f if f is None else round(f, 2)}   noflip {g if g is None else round(g, 2)}")
    print("\nnoflip length histogram:", N["length"]["histogram"])
    print("done", EXP / "scores" / "pilot_noflip.json")


if __name__ == "__main__":
    main()
