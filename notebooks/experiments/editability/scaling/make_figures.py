"""Build the canonical editability-scaling figures. One entry point, no duplicated plotting.

    python notebooks/experiments/editability/scaling/make_figures.py

Writes to `runs/scaling/figures/`:

    scaling_measured.png     only what has actually been run
    scaling_filled.png       the same layout with the empty grid cells FABRICATED, to judge style
    hyp_scale2edit.png       Sevan's Scale2edit: discworld rises with scale, like Othello
    hyp_insufficiency.png    Sevan's Insufficiency: discworld plateaus at ~0 forever
    hyp_peaked.png           the third outcome the current data already hints at

⚠ Everything except `scaling_measured.png` contains fabricated points. They are drawn hollow with
dashed connectors and carry a red banner. They exist to choose a visual style and to make the
competing hypotheses concrete — never to be read as results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]   # _HERE is already the scaling dir, so [3] is the repo — [4] escapes it
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pim.figures.scaling import grid  # noqa: E402

from collect import (  # noqa: E402
    ARCH_OURS, ARCH_THEIRS, PUBLISHED, SHOW_REFERENCE, collect,
)

OUT = _REPO / "runs" / "scaling" / "figures"
ARCHS = [ARCH_THEIRS, ARCH_OURS]
EDITORS = ["Nanda direction", "MLP grad steering", "PI injection (1 point)"]
VOLUMES = [100_000, 900_000, 20_000_000]
STEPS = [1.4e4, 5e4, 2.8e5, 1e6, 3e6]


def _mock(setting, arch, editor, games, steps, ei):
    return dict(setting=setting, arch=arch, editor=editor, games=int(games), steps=int(steps),
                edit_index=float(ei), measured=False, label="MOCK")


def _curve(lo, hi, k, n=len(STEPS)):
    """A saturating rise from `lo` toward `hi`; `k` sets how fast."""
    x = np.linspace(0, 1, n)
    return lo + (hi - lo) * (1 - np.exp(-k * x)) / (1 - np.exp(-k))


def fill(records, othello_ceiling, discworld_shape):
    """Fabricate the empty cells. `discworld_shape` in {'rise','flat','peak'}.

    Two fixes over the first draft, both from Sevan's review 2026-08-23:
    * Every editor reaches a comparable ceiling at scale. On the full OthelloGPT *all three* work,
      and PI injection is in fact the strongest (+0.697) while Nanda leads at low scale — so the
      mock has the editors CROSS rather than Nanda dominating everywhere.
    * A measured point must never float free of a line. Each (volume, steps) cell is mocked unless
      a measured point already occupies it, and the measured steps are merged into the step grid so
      the real points become nodes on the drawn curve.
    """
    out = list(records)
    for arch in ARCHS:
        cap = 1.0 if arch == ARCH_THEIRS else 0.45      # our 3.2M assumed capacity-limited
        for editor in EDITORS:
            # Nanda leads early and saturates; the gradient and PI editors start worse and
            # overtake at scale, which is the ordering the published model shows.
            early, late = {
                "Nanda direction": (0.55, 0.80),
                "PI injection (1 point)": (0.10, 1.00),
                "MLP grad steering": (0.05, 0.90),
            }[editor]
            # Mock EVERY volume that appears for this cell, measured ones included — otherwise a
            # measured point at 90k floats free while the mock line is drawn at 100k.
            vols = sorted({*VOLUMES, *[r["games"] for r in out
                                       if r["arch"] == arch and r["editor"] == editor]})
            for vi, games in enumerate(vols):
                vol = (vi + 1) / len(vols)
                have = [r for r in out if r["arch"] == arch and r["editor"] == editor
                        and r["games"] == games]
                steps = sorted({*STEPS, *[r["steps"] for r in have]})
                n = len(steps)
                w = np.linspace(0, 1, n)                       # 0 at the first step, 1 at the last
                gain = early + (late - early) * w              # editor ordering crosses over
                oth = _curve(-0.35, othello_ceiling * cap * vol, 3.0, n) * gain
                if discworld_shape == "rise":
                    dis = _curve(-0.55, 0.62 * cap * vol, 2.2, n) * gain
                elif discworld_shape == "flat":
                    dis = _curve(-0.55, 0.02, 2.5, n)
                else:                                          # peak then decline
                    dis = (_curve(-0.55, 0.45 * cap * vol, 4.0, n) * gain
                           - np.linspace(0, 0.5, n) ** 2 * 0.9 * vol)
                for s, o, d in zip(steps, oth, dis):
                    for setting, v in (("othello", o), ("discworld", d)):
                        if any(r["setting"] == setting and r["steps"] == s for r in have):
                            continue
                        out.append(_mock(setting, arch, editor, games, s, v))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = collect()
    print(f"{len(recs)} measured records")

    f = grid(recs, ARCHS, EDITORS, reference=PUBLISHED if SHOW_REFERENCE else None,
             suptitle="Editability vs training steps — MEASURED ONLY (2026-08-23)")
    f.savefig(OUT / "scaling_measured.png", dpi=150, bbox_inches="tight")

    for name, shape, title in (
        ("scaling_filled", "rise", "layout preview — empty cells FABRICATED"),
        ("hyp_scale2edit", "rise", "HYPOTHESIS · Scale2edit — discworld rises with scale"),
        ("hyp_insufficiency", "flat", "HYPOTHESIS · Insufficiency — discworld never leaves 0"),
        ("hyp_peaked", "peak", "HYPOTHESIS · Peaked — editability is a regime, not a limit"),
    ):
        f = grid(fill(recs, 0.75, shape), ARCHS, EDITORS, reference=PUBLISHED if SHOW_REFERENCE else None,
                 suptitle=f"Editability vs training steps — {title}", mock_note=True)
        f.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
        print(f"  wrote {name}.png")
    print(f"\nfigures in {OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
