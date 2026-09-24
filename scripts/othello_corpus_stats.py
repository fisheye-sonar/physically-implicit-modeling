#!/usr/bin/env python
"""Corpus statistics of an Othello instance → runs/_baselines/<instance>/corpus_stats.json.

    .pim/bin/python scripts/othello_corpus_stats.py                     # the four paper instances
    .pim/bin/python scripts/othello_corpus_stats.py --instance oth-adjacent-flip --n-games 2000

The numbers the paper quotes about the GAMES themselves (Setup → Othello variants): how often the
rules recolour a disc — discs flipped per move and per game (``pim.environments.othello.
counterfactual.flips_per_move``), measured by replaying the instance's held-out test games under
its own rules — and the game length. CPU only, about a minute per instance at 10,000 games.
Moved 2026-09-19 from ``experiments/adjacent_flip_ablation/scripts/pilot_adjacent_flip.py`` (which
measured freshly generated games before the corpus existed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.counterfactual import flips_per_move  # noqa: E402

PAPER_INSTANCES = ("oth-uniform", "oth-adjacent-flip", "oth-adjacent", "oth-noflip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", nargs="*", default=list(PAPER_INSTANCES))
    ap.add_argument("--n-games", type=int, default=10_000)
    a = ap.parse_args()
    for inst in a.instance:
        tok, ln = oc.load(
            oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",), instance=inst)[
                "test"
            ]
        )
        tok, ln = tok[: a.n_games], ln[: a.n_games]
        res = {
            "instance": inst,
            "split": "test",
            "rules": oc.rules_of(inst),
            "created": time.strftime("%Y-%m-%d %H:%M"),
            **flips_per_move(tok, ln, oc.rules_of(inst)),
            "game_length_mean": float(ln.mean()),
        }
        out = _REPO / "runs" / "_baselines" / inst / "corpus_stats.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, out)
        print(
            f"{inst}: {res['flips_per_move']:.3f} discs flipped per move ({res['flips_per_game']:.1f} per game, "
            f"{res['n_games']:,} games, mean length {res['game_length_mean']:.1f}) → {out.relative_to(_REPO)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
