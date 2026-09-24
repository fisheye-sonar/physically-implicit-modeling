"""Remove the CONTINUOUS-state inverse-map arms from every CATEGORICAL discworld block (2026-09-20, Sevan).

    .pim/bin/python experiments/categorical_inverse/scripts/clear_continuous_im.py            # DRY RUN: lists, writes nothing
    .pim/bin/python experiments/categorical_inverse/scripts/clear_continuous_im.py --apply    # rewrites the scores.json files

Until 2026-09-20 the scorer handed every categorical discworld block the inverse map of its BASIS — g fitted
on the continuous full state (position + velocity, 8 numbers) — and wrote g(continuous post-edit state) on the
categorical bench. The block's "IM" / "IM-NN" arms were therefore never an inversion of the categorical state
its probes read; they were the continuous editor evaluated on the categorical case selection. Sevan: those
numbers must never be reported as categorical IM — delete them.

For each ``runs/<topic>/<run>/scores.json`` of a discworld run, for each block with ``kind ==
"classification"`` whose ``inverse_map`` is not already the categorical one (``state`` ==
``pim.probes.inverse.CATEGORICAL_STATE``): the IM / IM-NN arms are removed from ``arms``, ``best["IM"]`` /
``best["IM-NN"]`` (and ``best_by_dims``) are set to null, ``inverse_map`` is removed, and
``inverse_cleared`` records the date and the reason. Nothing else in the file changes. Regression blocks and
Othello are untouched.

Nothing under runs/ is deleted: before the first rewrite of a file its current content is copied to
``scores_backup/scores_pre-categorical-inverse_<date>.json`` (kept if it already exists). Atomic write
(tmp → replace). Idempotent — a second run finds nothing. ⛔ Run it only while NO master_eval execution is
writing scores.json on this host (``ps -eo comm,args | awk '$1 ~ /^python/ && /nbconvert/ && /master_eval/'``);
the script refuses to --apply if it sees one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CATEGORICAL_STATE = "onehot-labels+cartesian-velocity"          # pim.probes.inverse.CATEGORICAL_STATE
WHY = ("the IM / IM-NN arms of this categorical block were the CONTINUOUS full-state inverse map of the block's basis "
       "scored on the categorical bench — not an inversion of the categorical state; removed 2026-09-20 (Sevan)")


def clear(scores: dict, today: str) -> list[str]:
    """Clear the continuous-state inverse arms of every categorical block of one scores dict IN PLACE.
    Returns the block keys changed."""
    if scores.get("env") != "discworld":
        return []
    done = []
    for key, blk in scores.get("bases", {}).items():
        if blk.get("kind") != "classification":
            continue
        if (blk.get("inverse_map") or {}).get("state") == CATEGORICAL_STATE:
            continue                                         # already the categorical map
        arms = blk.get("arms", [])
        has = any(a.get("editor") in ("IM", "IM-NN") for a in arms)
        if not has and not blk.get("inverse_map") and not (blk.get("best") or {}).get("IM"):
            continue
        blk["arms"] = [a for a in arms if a.get("editor") not in ("IM", "IM-NN")]
        for ed in ("IM", "IM-NN"):
            if "best" in blk:
                blk["best"][ed] = None
            for d in blk.get("best_by_dims", {}):
                blk["best_by_dims"][d][ed] = None
        blk.pop("inverse_map", None)
        blk["inverse_cleared"] = {"date": today, "why": WHY}
        done.append(key)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--runs-root", default=str(REPO / "runs"))
    a = ap.parse_args()
    if a.apply:
        ps = subprocess.run(["ps", "-eo", "comm,args"], capture_output=True, text=True).stdout
        if any(l.split()[0].startswith("python") and "nbconvert" in l and "master_eval" in l for l in ps.splitlines() if l.split()):
            sys.exit("a master_eval execution is running on this host — it rewrites scores.json files; try again when it ends")
    today = dt.date.today().isoformat()
    n_files = n_blocks = 0
    for sp in sorted(Path(a.runs_root).glob("*/*/scores.json")):
        scores = json.loads(sp.read_text())
        keys = clear(scores, today)
        if not keys:
            continue
        n_files += 1
        n_blocks += len(keys)
        print(f"{'CLEARED' if a.apply else 'would clear'}  {sp.parent.relative_to(a.runs_root)}  blocks {keys}")
        if a.apply:
            bdir = sp.parent / "scores_backup"
            bdir.mkdir(exist_ok=True)
            bak = bdir / f"scores_pre-categorical-inverse_{today}.json"
            if not bak.exists():
                shutil.copy2(sp, bak)
            tmp = sp.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(scores, indent=1, default=float))
            os.replace(tmp, sp)
    print(f"\n{'cleared' if a.apply else 'DRY RUN — would clear'} {n_blocks} categorical blocks in {n_files} runs"
          + ("" if a.apply else "   (re-run with --apply)"))


if __name__ == "__main__":
    main()
