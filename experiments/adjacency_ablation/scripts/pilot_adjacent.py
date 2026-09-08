"""Pilot gate for oth-adjacent (2026-09-08), BEFORE the 20M corpus — the no-flip lesson.

On 20k adjacency-rule games: lengths and passes; whether colour is a function of position
(the checkerboard theorem that voided oth-noflip); terminal-board diversity; legal-set size
by move and the exact Bayes CE; and the number that decides the bench — the fraction of
random (position, occupied non-centre tile) recolourings that CHANGE the legal set, with the
size of the differing set. Gate: >= --min-change of recolourings change the legal set.
Exit 0 = go, 2 = stop.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import CENTRE, synthetic_games  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

EXP = REPO / "experiments" / "adjacency_ablation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="oth-adjacent")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--n-recolour", type=int, default=4000)
    ap.add_argument("--min-change", type=float, default=0.30)
    a = ap.parse_args()
    t0 = time.time()
    rules = oc.rules_of(a.instance)
    games = synthetic_games(a.n, seed=0, n_workers=32, **rules)
    rng = np.random.default_rng(0)
    lengths = np.array([len(g) for g in games])
    n_pass = 0
    finals, black_by_sq = Counter(), np.zeros(64)
    parity_match = 0
    n_discs = 0
    legal_by_move = {}
    log_legal = []
    for g in games:
        b = OthelloBoardState(**rules)
        for t, mv in enumerate(g):
            legal = b.get_valid_moves()
            legal_by_move.setdefault(t, []).append(len(legal))
            log_legal.append(np.log(len(legal)))
            if b.tentative_move(mv) == 2:
                n_pass += 1
            b.umpire(mv)
        finals[b.state.tobytes()] += 1
        occ = b.state != 0
        black_by_sq += (b.state > 0).flatten()
        for sq in np.where(occ.flatten())[0]:
            parity_match += int(((sq // 8 + sq % 8) % 2 == 0) == (b.state[sq // 8, sq % 8] > 0))
        n_discs += int(occ.sum())
    black_frac = black_by_sq / a.n
    # recolouring test: does flipping one occupied non-centre tile change the legal set?
    changed, diff_sizes, empties = 0, [], 0
    for _ in range(a.n_recolour):
        g = games[rng.integers(a.n)]
        L = int(rng.integers(5, min(31, len(g))))
        b = OthelloBoardState(**rules)
        b.update(g[:L], prt=False)
        pre = set(b.get_valid_moves())
        occ = [sq for sq in range(64) if b.state[sq // 8, sq % 8] != 0 and sq not in CENTRE]
        sq = int(rng.choice(occ))
        b.state[sq // 8, sq % 8] *= -1
        post = set(b.get_valid_moves())
        if not post:
            empties += 1
        elif post != pre:
            changed += 1
            diff_sizes.append(len(pre ^ post))
    rep = {
        "instance": a.instance, "rules": rules, "n_games": a.n,
        "length": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max()),
                   "frac_60": float((lengths == 60).mean())},
        "passes_per_game": n_pass / a.n,
        "distinct_terminal_boards": len(finals), "most_common_terminal_share": finals.most_common(1)[0][1] / a.n,
        "colour_eq_parity_frac": parity_match / n_discs,
        "black_frac_by_square": {"min": float(black_frac.min()), "max": float(black_frac.max()),
                                 "std": float(black_frac.std())},
        "legal_size_by_move": {int(t): float(np.mean(v)) for t, v in legal_by_move.items() if t % 5 == 0},
        "bayes_ce": float(np.mean(log_legal)),
        "recolour": {"n": a.n_recolour, "frac_changed": changed / a.n_recolour,
                     "frac_emptied": empties / a.n_recolour,
                     "diff_size_hist": {int(k): int(v) for k, v in sorted(Counter(diff_sizes).items())},
                     "diff_size_mean": float(np.mean(diff_sizes)) if diff_sizes else 0.0},
        "minutes": (time.time() - t0) / 60,
    }
    (EXP / "scores").mkdir(parents=True, exist_ok=True)
    (EXP / "scores" / "pilot_adjacent.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = rep["recolour"]["frac_changed"] >= a.min_change and rep["colour_eq_parity_frac"] < 0.9 \
        and rep["distinct_terminal_boards"] > 0.5 * a.n
    print(f"\nGATE {'PASS' if ok else 'FAIL'}: recolourings changing the legal set "
          f"{rep['recolour']['frac_changed']:.1%} (need >= {a.min_change:.0%}); colour==parity "
          f"{rep['colour_eq_parity_frac']:.3f}; distinct terminal boards {rep['distinct_terminal_boards']}/{a.n}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
