#!/usr/bin/env python
"""Synthesise Li-style intervention cases at EVERY move number (1…59).

Li et al.'s shipped benchmark covers prefixes of length 5–30 only. To read editability by
game position over the whole game, this builds new cases with the same recipe, as
measured on the shipped 1001 (2026-09-02): a real game history plus ONE occupied,
non-centre square flipped to the opposite colour, where the flip changes the legal-move
set (1001/1001 shipped cases do) and never empties it (0/1001).

Games come from the canonical corpus's TEST split (uniform-random-legal generator, index
range [90M, 90M+10k) — held out from training and disjoint from the probe games). For
each move number t a seeded sample of games with at least t moves gives the length-t
prefix; a uniformly random qualifying square is flipped; a flip whose legal set is
unchanged or empty is rejected and another square tried, then another game.

Output: cases/<name>.pkl — a list of {history, pos_int, ori_color, game} dicts, the
shipped pkl's own format (plus the source game index), so
`pim.environments.othello.bench.benchmark_from_cases` consumes it unchanged — and
cases/<name>.json with the recipe, seed, per-step counts and rejection statistics.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.data import canonical_vocab  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402

CENTRE = (27, 28, 35, 36)          # never flipped in the shipped benchmark (0/1001)


def legal_moves(history: list[int]) -> tuple[OthelloBoardState, list[int]]:
    b = OthelloBoardState()
    b.update(history, prt=False)
    return b, sorted(b.get_valid_moves())


def legal_after_flip(history: list[int], sq: int, ori_color: float) -> list[int]:
    post = OthelloBoardState()
    post.update(history, prt=False)
    post.state[sq // 8, sq % 8] = int(2 - ori_color) - 1     # the benchmark's own flip
    return sorted(post.get_valid_moves())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-step", type=int, default=256)
    ap.add_argument("--steps", type=int, nargs=2, default=[1, 59], metavar=("FIRST", "LAST"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"synth_seed{a.seed}_n{a.per_step}"
    out_dir = REPO / "experiments" / "othello_edit_by_step" / "cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",))["test"])
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    print(f"test split: {len(hist)} games, lengths {min(map(len, hist))}–{max(map(len, hist))}",
          flush=True)

    rng = np.random.default_rng(a.seed)
    cases, stats = [], {}
    for t in range(a.steps[0], a.steps[1] + 1):
        pool = np.array([i for i, h in enumerate(hist) if len(h) >= t])
        order = rng.permutation(pool)
        got = tried = rej_same = rej_empty = rej_no_pre = 0
        for g in order:
            if got >= a.per_step:
                break
            tried += 1
            h = hist[g][:t]
            board, pre = legal_moves(h)
            if not pre:                                  # no continuation to predict
                rej_no_pre += 1
                continue
            occ = [sq for sq in range(64)
                   if board.state[sq // 8, sq % 8] != 0 and sq not in CENTRE]
            for sq in rng.permutation(occ):
                sq = int(sq)
                ori = 0.0 if board.state[sq // 8, sq % 8] < 0 else 2.0
                post = legal_after_flip(h, sq, ori)
                if not post:
                    rej_empty += 1
                    continue
                if post == pre:
                    rej_same += 1
                    continue
                cases.append({"history": list(h), "pos_int": sq, "ori_color": ori,
                              "game": int(g)})
                got += 1
                break
        stats[t] = {"n": got, "games_tried": tried, "pool": int(len(pool)),
                    "rejected_same_legal": rej_same, "rejected_empty_legal": rej_empty,
                    "rejected_no_legal_pre": rej_no_pre}
        print(f"move {t:2d}: {got} cases from {tried} games (pool {len(pool)}; rejected "
              f"same-legal {rej_same}, empty {rej_empty}, no-pre {rej_no_pre})", flush=True)

    with open(out_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(cases, f)
    manifest = {"name": name, "seed": a.seed, "per_step": a.per_step, "steps": a.steps,
                "n_cases": len(cases), "source": "oth-uniform corpus TEST split "
                f"(oc.build(LADDER['D'])['test'], {len(hist)} games)",
                "recipe": "length-t prefix of a held-out game; one uniformly random "
                          "occupied non-centre square flipped to the opposite colour; "
                          "rejected if the legal set is unchanged or empty",
                "stats": stats, "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / f"{name}.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(cases)} cases -> {out_dir / (name + '.pkl')}  "
          f"[{manifest['minutes']} min]")


if __name__ == "__main__":
    main()
