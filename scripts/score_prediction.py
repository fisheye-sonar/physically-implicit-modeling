#!/usr/bin/env python
"""Fold the ``prediction`` block (held-out predictive loss) into scored runs' scores.json.

    .pim/bin/python scripts/score_prediction.py                          # every scored run that lacks it
    .pim/bin/python scripts/score_prediction.py --only L-dw-8ray-20m L-dw-8ray-tok-20m

The block is ``pim.environments.prediction.score_run`` — the run's objective on its instance's
held-out split, per sequence, plus (frames-as-tokens runs) both readings of the token model.
Like the scorer's other fold-ins it ADDS a block to an already-scored run and touches nothing
else: one dated backup under ``runs/<run>/scores_backup/``, atomic replace, a run already at
``PRED_VERSION`` is skipped. Quarantined (``_``-prefixed) and archived runs are never touched;
``PIM_SKIP_TOPICS`` (comma-separated, default ``training_curve``) names topics to leave out.

⛔ Do not run this while ``master_eval`` is executing on the same machine: both rewrite
scores.json files. Seconds per run on a GPU (one forward pass over 10k held-out sequences);
Othello runs need no model at all (the block is read from their ``gates``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments.prediction import PRED_VERSION, score_run  # noqa: E402


def counted(rel: Path, skip_topics: set[str]) -> bool:
    return (rel.parts[0] != "archive" and rel.parts[0] not in skip_topics
            and not any(p.startswith("_") for p in rel.parts))


def write_scores(sp: Path, scores: dict) -> None:
    bdir = sp.parent / "scores_backup"
    bdir.mkdir(exist_ok=True)
    tag = f"scores_prediction-{PRED_VERSION}_"
    if not any(p.name.startswith(tag) for p in bdir.glob("scores_*.json")):
        shutil.copy2(sp, bdir / f"{tag}{dt.date.today().isoformat()}.json")
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scores, indent=1, default=float))
    os.replace(tmp, sp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="run directory names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    skip = {t for t in os.environ.get("PIM_SKIP_TOPICS", "training_curve").split(",") if t}
    for sp in sorted((_REPO / "runs").rglob("scores.json")):
        rel = sp.relative_to(_REPO / "runs")
        if not counted(rel, skip) or (a.only and sp.parent.name not in a.only):
            continue
        s = json.loads(sp.read_text())
        if s.get("prediction", {}).get("version") == PRED_VERSION and not a.force:
            continue
        t0 = time.time()
        try:
            block = score_run(sp.parent, device=a.device)
        except Exception as e:                                   # one bad run never stops the pass
            print(f"  {rel.parent}: SKIPPED — {type(e).__name__}: {e}", flush=True)
            continue
        s = json.loads(sp.read_text())                           # re-read: keep the write window short
        s["prediction"] = block
        write_scores(sp, s)
        line = "  ".join(f"{k} {v['loss']:.5f}" for k, v in block["readings"].items())
        print(f"  {rel.parent}: {line}  [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
