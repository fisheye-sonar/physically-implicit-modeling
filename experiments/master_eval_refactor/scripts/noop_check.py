"""No-op check: over the whole runs/ tree, the refactored driver must find NOTHING to do.

Runs the real `score_all_baselines` and `score_all` loops with dry_run=True (every decision is
the production code path; nothing is fitted, scored or written) under the environment the
paper_ci queue scores with. Exit code 0 iff both to-do lists are empty.

    PIM_DW_BASES=frustum,cartesian PIM_SKIP_TOPICS=training_curve \
        .pim/bin/python experiments/master_eval_refactor/scripts/noop_check.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import REPO, settings  # noqa: E402

os.chdir(REPO)
from pim.scoring import scan_runs, score_all, score_all_baselines  # noqa: E402

SETTINGS, eval_version = settings()
print(f"env: PIM_DW_BASES={os.environ.get('PIM_DW_BASES')!r} PIM_SKIP_TOPICS={os.environ.get('PIM_SKIP_TOPICS')!r} "
      f"PIM_ONLY_RUNS={os.environ.get('PIM_ONLY_RUNS')!r}  → dw_bases {SETTINGS['dw_bases']}")
RUNS = scan_runs()
print(f"{len(RUNS)} runs scanned ({sum(r['scored'] for r in RUNS)} with a scores.json)\n")

todo_b = score_all_baselines(RUNS, SETTINGS, dry_run=True)
print()
todo_r = score_all(RUNS, SETTINGS, eval_version, dry_run=True)

out = REPO / "experiments" / "master_eval_refactor" / "scores" / "noop_check.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"env": {k: os.environ.get(k) for k in ("PIM_DW_BASES", "PIM_SKIP_TOPICS", "PIM_ONLY_RUNS")},
                           "n_runs": len(RUNS), "baselines_todo": todo_b, "runs_todo": todo_r}, indent=1, default=str))
print(f"\n=== NO-OP CHECK: {len(RUNS)} runs · baselines to fit: {len(todo_b)} · runs to score / add to: {len(todo_r)} ===")
for t in todo_b + todo_r:
    print("   ", t)
sys.exit(0 if not todo_b and not todo_r else 1)
