"""Assemble pim/scoring/ from master_eval's cell sources BY SLICING, never by retyping.

Every function body lands verbatim; the only textual edits are the ones listed in EDITS
(globals -> parameters / imports) and they are applied by exact-match replacement, each
asserted to hit exactly the expected number of times.
"""
import re
import sys
from pathlib import Path

CELLS = Path(sys.argv[1])
OUT = Path(sys.argv[2]) / "pim" / "scoring"
OUT.mkdir(parents=True, exist_ok=True)
C = {i: (CELLS / f"cell{i}.py").read_text() for i in range(1, 9)}


def between(src: str, start: str, end: str | None) -> str:
    """The text from the line starting with `start` up to (not including) the line starting with `end`."""
    i = src.index(start)
    assert i == 0 or src[i - 1] == "\n", start
    j = len(src) if end is None else src.index("\n" + end, i) + 1
    return src[i:j].rstrip("\n") + "\n"


def header_comment(src: str) -> str:
    """The leading comment block of a cell, kept as comments."""
    lines = []
    for ln in src.splitlines():
        if ln.startswith("#"):
            lines.append(ln)
        else:
            break
    return "\n".join(lines) + "\n"


def sub(text: str, old: str, new: str, count: int) -> str:
    n = text.count(old)
    assert n == count, f"expected {count} x {old!r}, found {n}"
    return text.replace(old, new)


# ── runs.py ── cell [1] ───────────────────────────────────────────────────────
runs = '''"""Which runs the canonical scorer sees (moved verbatim from master_eval cell [1], 2026-09-19)."""
''' + header_comment(C[1]) + '''import json
import os
import subprocess
from pathlib import Path

import torch

from pim.environments import layout

REPO = layout.REPO
DEV = "cuda" if torch.cuda.is_available() else "cpu"


''' + between(C[6], "def _sha", "def _pack") + "\n" + between(C[1], "def training_complete", "RUNS = scan_runs()")
(OUT / "runs.py").write_text(runs.rstrip("\n") + "\n")

# ── blocks.py ── cell [3] schema half + othello_blocks from cell [4] ──────────
blocks = '''"""The scores.json BLOCK: which blocks a run gets, and the shape of one (moved verbatim from
master_eval cells [3] / [4], 2026-09-19). `probe_block` is the on-disk contract every reader of a
scores.json depends on (pim.figures.tables, the paper_ci ledger, scripts/score_prediction.py)."""
''' + header_comment(C[3]) + '''import json

import numpy as np

from pim.environments.discworld.grid_target import categorical_target, snapped_target
from pim.metrics.decodability import probe_skill_from_stats
from pim.scoring.runs import REPO


''' + between(C[3], "def best_arm", "def inverse_discworld") + "\n" + between(C[5], "def othello_blocks", "def othello_arms")
blocks = sub(blocks, "s=SETTINGS", "s", 5)
(OUT / "blocks.py").write_text(blocks.rstrip("\n") + "\n")

# ── discworld.py ── cell [3] scorer half + cell [3b] ──────────────────────────
dw = '''"""The discworld scorers — frame models and frames-as-tokens models (moved verbatim from
master_eval cells [3] / [3b], 2026-09-19). Thin wiring over pim.environments.discworld."""
''' + header_comment(C[4]) + '''import json

from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld import token_bench as tkb
from pim.environments.discworld.tokens import FrameVocab
from pim.models import n_points
from pim.probes.mlp import check_probe_sanity
from pim.scoring.blocks import attach_inverse, discworld_blocks, dw_block_setup, probe_block
from pim.scoring.runs import REPO


''' + between(C[3], "def inverse_discworld", 'print("discworld scorer ready') + "\n" + \
    between(C[4], "def score_discworld_tokens", 'print("discworld TOKEN scorer ready')
dw = sub(dw, "s=SETTINGS", "s", 3)
(OUT / "discworld.py").write_text(dw.rstrip("\n") + "\n")

# ── othello.py ── cell [4] ────────────────────────────────────────────────────
oth = '''"""The Othello scorer (moved verbatim from master_eval cell [4], 2026-09-19). Thin wiring over
pim.environments.othello."""
''' + header_comment(C[5]) + '''import json

from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import case_targets, load_benchmark
from pim.environments.othello.data import tokens_and_labels, canonical_vocab
from pim.metrics.decodability import probe_skill_from_stats
from pim.metrics.set_editability import move_fidelity_ratio
from pim.models import n_points
from pim.probes.mlp import check_probe_sanity
from pim.scoring.blocks import EDITORS_SCORED, IM_VERSION, probe_block

''' + between(C[5], "PROBE_SOURCES = {", "def othello_blocks") + "\n" + \
    between(C[5], "def othello_arms", 'print("othello scorer ready')
oth = sub(oth, "s=SETTINGS", "s", 2)
(OUT / "othello.py").write_text(oth.rstrip("\n") + "\n")

# ── baselines.py ── cell [5] ──────────────────────────────────────────────────
funcs = between(C[6], "BASELINES_DIR =", "def _sha") + "\n" + between(C[6], "def _pack", "# every (instance, arch) pair")
loop = between(C[6], "# every (instance, arch) pair", 'print("\\nall baselines present")')
loop = "".join(("    " + ln if ln.strip() else ln) for ln in loop.splitlines(keepends=True))
base = '''"""The two decodability floors per (instance, architecture) (moved verbatim from master_eval
cell [5], 2026-09-19). `score_all_baselines` is the cell's top-level loop."""
''' + header_comment(C[6]) + '''import json
import time

import torch

from pim.environments import layout                 # every datasets/ path (layout v2)
from pim.environments.discworld import arms as dwa
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.metrics.decodability import insample_gap_from_stats, probe_skill_from_stats
from pim.models import load_checkpoint
from pim.probes.baselines import random_init_model
from pim.scoring.blocks import by_point, dw_bases_for
from pim.scoring.othello import _probe_games
from pim.scoring.runs import DEV, REPO, _sha

''' + funcs + '''

def score_all_baselines(runs, s, dry_run=False) -> list:
    """Fill in every floor a baselines.json lacks, for every (instance, arch) among `runs`.
    `dry_run` reports what WOULD be fitted and fits nothing; returns that to-do list."""
    todo_all = []
''' + loop + '''    print("\\nall baselines present")
    return todo_all
'''
base = sub(base, "s=SETTINGS", "s", 6)      # 5 defs + the loop's dw_bases_for(inst, s=SETTINGS) keyword
base = sub(base, "dw_bases_for(inst, s)", "dw_bases_for(inst, s)", 3)   # (assert: every call passes s)
(OUT / "baselines.py").write_text(base.rstrip("\n") + "\n")

# ── driver.py ── cell [6] ─────────────────────────────────────────────────────
helpers = between(C[7], "def scorer_for", "for r in RUNS:")
loop = between(C[7], "for r in RUNS:", 'print("\\nall runs scored")')
loop = "".join(("    " + ln if ln.strip() else ln) for ln in loop.splitlines(keepends=True))
drv = '''"""Score every run that needs it, adding only what is missing (moved verbatim from master_eval
cell [6], 2026-09-19). `score_all` is the cell's top-level loop."""
''' + header_comment(C[7]) + '''import datetime as _dt
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld import token_bench as tkb
from pim.environments.discworld.tokens import FrameVocab
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import load_benchmark
from pim.models import load_checkpoint, n_points
from pim.scoring.blocks import IM_VERSION, best_arm, discworld_blocks, othello_blocks
from pim.scoring.discworld import inverse_discworld, score_discworld, score_discworld_tokens
from pim.scoring.othello import _probe_games, score_othello
from pim.scoring.runs import DEV, REPO, _sha

''' + helpers + '''

def score_all(runs, s, eval_version, dry_run=False) -> list:
    """Score every run in `runs` whose scores.json is missing, stale, or lacks a block the
    settings ask of it. `eval_version(r)` is the notebook's version rule. `dry_run` reports
    what WOULD be done and does nothing; returns that to-do list."""
    todo_all = []
''' + loop + '''    print("\\nall runs scored")
    return todo_all
'''
drv = sub(drv, "s=SETTINGS", "s", 1)
drv = sub(drv, "for r in RUNS:", "for r in runs:", 1)
(OUT / "driver.py").write_text(drv.rstrip("\n") + "\n")

# ── summary.py ── cell [7] ────────────────────────────────────────────────────
body = between(C[8], "for r in RUNS:", None)
body = "".join(("    " + ln if ln.strip() else ln) for ln in body.splitlines(keepends=True))
summ = '''"""Per-run summaries: the headline block of each scores.json, human-readable (moved verbatim
from master_eval cell [7], 2026-09-19)."""
''' + header_comment(C[8]) + '''import json


def print_summaries(runs) -> None:
''' + sub(body, "for r in RUNS:", "for r in runs:", 1)
(OUT / "summary.py").write_text(summ.rstrip("\n") + "\n")

for p in sorted(OUT.glob("*.py")):
    print(f"{p.name:14s} {len(p.read_text().splitlines()):4d} lines")
