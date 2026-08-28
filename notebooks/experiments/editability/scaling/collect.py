"""Assemble editability-scaling records from the result JSONs on disk.

The single place that knows where each number lives and how to normalise it. Everything downstream
(`pim.figures.scaling`, any notebook) consumes the record list this produces, so a number is
defined once and a new run is added by extending `RUNS` below — never by re-deriving a metric.

Normalisation decisions, all deliberate:

* **Edit Index is the absolute post-edit value**, best over every (alpha, layer, probe-target)
  choice for that editor. Best-of is honest here because the question a panel answers is "what is
  this editor's ceiling on this model", not "does an arbitrary setting work".
* **Steps, not epochs.** An epoch is a different amount of compute at every data volume.
* **Editor names are unified across settings.** `linear_intervention`'s Othello names and
  `editability.py`'s discworld names differ; the map below is the only place that is reconciled.

⚠ The two settings' Edit Index share a name, a range and a sign convention but **not a
construction** — ray-RMSE over the observation (discworld) vs a move distribution over 64 squares
against uniform-over-legal (Othello). Curves are comparable in shape and in whether they cross
zero; individual cells are not comparable across settings.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
ARCH_OURS = "Transformer S (3.2M params, 4 blocks)"
ARCH_THEIRS = "Transformer L (25.3M params, 8 blocks)"

# editor label used in the figures  <-  the names the two evaluators emit
EDITOR_MAP = {
    "MLP grad steering": ("MLP grad steering",),
    "PI injection (1 point)": ("PI injection (1 point)",),
    "Nanda direction": ("Nanda addition", "Nanda (target-current)"),
}

# run-name -> (setting, arch, unique games, total optimiser steps)
RUNS = {
    "ENVCTRL_theirs_900k":   ("othello",   ARCH_THEIRS,   900_000,  14_064),
    "ENVCTRL14_theirs_900k": ("othello",   ARCH_THEIRS,   900_000,  49_224),
    "L90_theirs_othello":    ("othello",   ARCH_THEIRS,   900_000, 284_850),
    "A_pilot_900k":          ("discworld", ARCH_THEIRS,   900_000,  12_660),
    "A_pilot14_900k":        ("discworld", ARCH_THEIRS,   900_000,  44_296),
    "L90_theirs_discworld":  ("discworld", ARCH_THEIRS,   900_000, 284_760),
}

# Our architecture on Othello — the ladder, from `ours_on_othello/ladder_edit_full.json`.
# That file keys on run name; steps come from each run's own config.
OURS_LADDER = {
    "M_w16":  ("othello", ARCH_OURS,     90_000,  95_100),
    "L1_w16": ("othello", ARCH_OURS,  1_000_000,  95_100),
    "L2_w16": ("othello", ARCH_OURS,  5_000_000,  95_100),
    "D_w16":  ("othello", ARCH_OURS, 20_000_000,  95_100),
}

# Our architecture on discworld — `W16`, the thread's reference model. Best over every editor and
# every (layer, alpha) tried, from `pinv_alpha_discworld.py` and `transformer_world_state.ipynb`.
OURS_DISCWORLD = [
    ("MLP grad steering",     -0.194),   # othello_gpt/, best probe-derived write, 2026-08-18
    ("PI injection (1 point)", -0.443),  # pinv_alpha_discworld.py, best (point, alpha), 2026-08-21
    ("Nanda direction",       -0.118),   # nanda_on_discworld.py, best alpha, 2026-08-21
]


def _best(arms, editor_names, key) -> float | None:
    vals = [a[key] for a in arms if a.get("editor") in editor_names]
    return max(vals) if vals else None


def collect() -> list[dict]:
    """Every measured point we have, as figure records."""
    out: list[dict] = []
    arch_runs = REPO / "runs" / "othello_arch"

    for run, (setting, arch, games, steps) in RUNS.items():
        p = arch_runs / f"{run}_editability.json"
        if not p.exists():
            continue
        blob = json.loads(p.read_text())
        for label, names in EDITOR_MAP.items():
            if setting == "othello":
                v = _best(blob["arms"], names, "edit_index_union")
            else:  # discworld: arms live under each probe target
                vals = [x for t in blob["targets"].values()
                        for x in [_best(t["arms"], names, "edit_index")] if x is not None]
                v = max(vals) if vals else None
            if v is not None:
                out.append(dict(setting=setting, arch=arch, editor=label, games=games,
                                steps=steps, edit_index=float(v), measured=True, label=run))

    # our architecture on Othello (the ladder) — Nanda + PI only; no gradient arm was swept there
    lad = REPO / "runs" / "ours_on_othello" / "ladder_edit_full.json"
    if lad.exists():
        blob = json.loads(lad.read_text())
        for run, (setting, arch, games, steps) in OURS_LADDER.items():
            if run not in blob:
                continue
            # `ladder_edit_full.json` stores one best-by-Edit-Index arm per run, whose `arm`
            # field names which editor produced it. Attribute the point to that editor only.
            best = blob[run].get("best_ei", {})
            arm = str(best.get("arm", ""))
            label = ("PI injection (1 point)" if arm.startswith("pinv")
                     else "Nanda direction" if arm in ("add", "t-c") else None)
            if label is None or not best:
                continue
            out.append(dict(setting=setting, arch=arch, editor=label, games=games,
                            steps=steps, edit_index=float(best["edit_index_union"]),
                            measured=True, label=run))

    for label, v in OURS_DISCWORLD:
        out.append(dict(setting="discworld", arch=ARCH_OURS, editor=label, games=90_000,
                        steps=95_100, edit_index=float(v), measured=True, label="W16"))
    return out


# Toggle with SHOW_REFERENCE. It is a DIFFERENT data scale (20M games, fully trained), so it
# anchors the top of the axis but should not be read as a target for any cell on these plots.
SHOW_REFERENCE = True
PUBLISHED = {"OthelloGPT (20M games, 25.3M params)": 0.697}


if __name__ == "__main__":
    recs = collect()
    print(f"{len(recs)} measured records\n")
    for r in sorted(recs, key=lambda r: (r["arch"], r["editor"], r["setting"], r["steps"])):
        print(f"  {r['arch']:<22} {r['editor']:<24} {r['setting']:<10} "
              f"{r['games']:>10,} games {r['steps']:>8,} steps  EI {r['edit_index']:+.4f}")
