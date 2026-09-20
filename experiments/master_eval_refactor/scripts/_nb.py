"""Shared by the gate scripts: the notebook's OWN settings cell, and the pre-refactor cells.

`settings()` executes cell [2] of notebooks/master_eval.ipynb as it is on disk, so the gate
scores with exactly the SETTINGS / eval_version production will use (never a copy).
`old_namespace()` executes the pre-refactor notebook's function definitions (from git, commit
OLD_COMMIT) WITHOUT its two top-level loops, so the old scorers can be called side by side.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OLD_COMMIT = "c023401"          # the last commit with the scorers inside the notebook


def _cells(nb_json: str) -> list[str]:
    return ["".join(c["source"]) for c in json.loads(nb_json)["cells"] if c["cell_type"] == "code"]


def settings() -> tuple[dict, callable]:
    """(SETTINGS, eval_version) from the notebook on disk — cell [2], executed."""
    cells = _cells((REPO / "notebooks" / "master_eval.ipynb").read_text())
    src = next(c for c in cells if c.startswith("# [2] The canonical evaluation settings"))
    ns = {"os": os}
    exec(compile(src, "master_eval[2]", "exec"), ns)
    return ns["SETTINGS"], ns["eval_version"]


def old_namespace() -> dict:
    """The pre-refactor notebook's definitions, loops cut off. Cell [1] runs its (read-only) scan."""
    nb = subprocess.run(["git", "show", f"{OLD_COMMIT}:notebooks/master_eval.ipynb"], cwd=REPO,
                        capture_output=True, text=True, check=True).stdout
    cells = _cells(nb)
    assert len(cells) == 8, len(cells)
    cuts = {5: "# every (instance, arch) pair that has a trained run", 6: "for r in RUNS:", 7: None}
    ns: dict = {"__name__": "old_master_eval"}
    os.chdir(REPO / "notebooks")             # cell [1] derives REPO from the working directory
    for i, src in enumerate(cells):
        if i in cuts:
            if cuts[i] is None:
                continue                      # the summaries cell: nothing but a loop
            src = src[: src.index(cuts[i])]
        exec(compile(src, f"old_master_eval[{i}]", "exec"), ns)
    assert Path(ns["REPO"]) == REPO, (ns["REPO"], REPO)
    return ns


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
