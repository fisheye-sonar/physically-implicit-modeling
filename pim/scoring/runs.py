"""Which runs the canonical scorer sees (moved verbatim from master_eval cell [1], 2026-09-19)."""
# [1] Scan runs/ (recursively) -> the run table. Two exclusions, both by convention:
#     runs/archive/ (hardcoded, per the housecleaning rules) and any topic dir starting
#     with "_" (private/scratch, e.g. runs/_smoke — score those by renaming the topic).
#     A run STILL TRAINING is listed but not scored (2026-09-05): best_model.pt appears at
#     the first validation pass, so without this check an executing notebook would score
#     a 5k-step checkpoint, stamp it with the current EVAL_VERSION, and the finished model
#     would then be skipped as "already scored". Complete = the last logged step in
#     metrics.jsonl reached config train.steps.
import json
import os
import subprocess
from pathlib import Path

import torch

from pim.environments import layout

REPO = layout.REPO
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def training_complete(run_dir: Path, cfg: dict) -> bool:
    target = int(cfg.get("train", {}).get("steps", 0) or 0)
    mp = run_dir / "metrics.jsonl"
    if not target or not mp.exists():
        return True                                      # nothing to compare against
    last = 0
    for line in mp.read_text().splitlines():
        if line.strip():
            last = max(last, int(json.loads(line).get("step", 0)))
    return last >= target

def scan_runs(root: Path = REPO / "runs") -> list[dict]:
    rows = []
    for cfg_path in sorted(root.rglob("config.json")):
        rel = cfg_path.relative_to(root)
        if rel.parts[0] == "archive" or rel.parts[0].startswith("_"):
            continue
        cfg = json.loads(cfg_path.read_text())
        run_dir = cfg_path.parent
        if not (run_dir / "best_model.pt").exists():
            continue                                     # not a run dir (stray config)
        only = os.environ.get("PIM_ONLY_RUNS", "")        # smoke hook (2026-09-15): a comma list of run names
        if only and run_dir.name not in only.split(","):
            continue
        skip_topics = os.environ.get("PIM_SKIP_TOPICS", "")   # a comma list of topics to leave alone (2026-09-15, Sevan:
        if skip_topics and rel.parts[0] in skip_topics.split(","):   # training_curve is not in the paper — no IM scoring there)
            continue
        if not training_complete(run_dir, cfg):
            print(f"skip  {rel.parts[0]}/{run_dir.name}  (still training)")
            continue
        rows.append({
            "topic": str(rel.parts[0]), "run": run_dir.name, "dir": run_dir,
            "arch": cfg.get("arch", "?"),
            "env": cfg.get("data", {}).get("env", "?"),
            "instance": cfg.get("data", {}).get("instance", "?"),
            "n_params": cfg.get("n_params"),
            "scored": (run_dir / "scores.json").exists(),
        })
    return rows
