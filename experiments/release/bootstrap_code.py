"""Bootstrap the release tree: copy the in-scope code into RELEASE with the mechanical renames.

Run once, before the release workers prune and rewrite. Idempotent only on an empty target
(refuses to overwrite existing files). See SPEC.md for the naming map.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

PRIVATE = Path("/home/sevan/research/PIM/physically-implicit-modeling")
RELEASE = Path("/home/sevan/research/PIM/generative-models-as-simulators")

SCRIPTS = ["train.py", "generate_dataset.py", "make_othello_corpus.py", "make_othello_edits.py",
           "make_edit_selection.py", "make_discworld_tokens.py", "fit_probes.py",
           "layout_checkpoint_replicate.py", "score_prediction.py", "bayes_floor.py",
           "reachability_table.py", "two_flip_editability.py", "othello_corpus_stats.py",
           "demos/demo.py", "demos/play.py"]
SKIP_FILES = {"intervention_benchmark.pkl"}

RUNS = [("L-oth-adjacent-flip-20m", "othello/adjacent-flip"),
        ("L-oth-adjacent-20m", "othello/adjacent-noflip"),
        ("L-oth-noflip-20m", "othello/standard-noflip"),
        ("L-oth-20m", "othello/standard"),
        ("L-dw-8ray-obs5-20m", "rayworld/obs5"),
        ("L-dw-8ray-tok-20m", "rayworld/8-ray-tokens"),
        ("L-dw-noiseless-20m", "rayworld/standard"),
        ("L-dw-blink-20m", "rayworld/blink"),
        ("L-dw-128ray-20m", "rayworld/128-ray"),
        ("L-dw-16ray-20m", "rayworld/16-ray"),
        ("L-dw-8ray-20m", "rayworld/8-ray"),
        ("L-dw-5ray-20m", "rayworld/5-ray"),
        ("L-dw-smooth-20m", "rayworld/smooth")]
INSTANCES = [(r"oth-adjacent-flip", "adjacent-flip"),
             (r"oth-adjacent(?![-\w])", "adjacent-noflip"),
             (r"oth-noflip", "standard-noflip"),
             (r"oth-uniform", "standard"),
             (r"dw-8ray-obs5", "obs5"),
             (r"dw-noiseless", "standard"),
             (r"dw-blink", "blink"),
             (r"dw-128ray", "128-ray"),
             (r"dw-16ray", "16-ray"),
             (r"dw-8ray(?![-\w])", "8-ray"),
             (r"dw-5ray", "5-ray"),
             (r"dw-smooth", "smooth")]
IDENTS = [(r"discworld", "rayworld"), (r"Discworld", "Rayworld"), (r"DiscWorld", "Rayworld"),
          (r"DISCWORLD", "RAYWORLD"),
          (r"\bdw_", "rw_"), (r"_dw\b", "_rw"), (r"\bDW_", "RW_"), (r"_DW\b", "_RW"),
          (r"\bdwa\b", "rwa"), (r"\bdwb\b", "rwb"), (r"\bdw\b", "rw"), (r"\bDW\b", "RW")]


def transform(text: str) -> str:
    for old, new in RUNS:
        text = text.replace(old, new)
    for pat, new in INSTANCES:
        text = re.sub(pat, new, text)
    for pat, new in IDENTS:
        text = re.sub(pat, new, text)
    return text


def new_rel(rel: Path) -> Path:
    s = rel.as_posix()
    s = s.replace("environments/discworld", "environments/rayworld")
    s = s.replace("scoring/discworld.py", "scoring/rayworld.py")
    s = s.replace("make_discworld_tokens.py", "make_rayworld_tokens.py")
    return Path(s)


def copy_one(src: Path, rel: Path) -> None:
    dst = RELEASE / new_rel(rel)
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in {".py", ".md", ".txt", ".toml", ".cfg"}:
        dst.write_text(transform(src.read_text()))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    n = 0
    for src in sorted((PRIVATE / "pim").rglob("*")):
        if src.is_dir() or "__pycache__" in src.parts or src.suffix == ".pyc" or src.name in SKIP_FILES:
            continue
        copy_one(src, src.relative_to(PRIVATE))
        n += 1
    for s in SCRIPTS:
        copy_one(PRIVATE / "scripts" / s, Path("scripts") / s)
        n += 1
    print(f"copied {n} files into {RELEASE}")


if __name__ == "__main__":
    main()
