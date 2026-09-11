"""Haufe-corrected editing on L-oth-adjacent-flip-20m (2026-09-11) — the Result-4 test of
`research/findings/edit-direction-alignment.md` applied to the new instance. Thin caller of the
existing `experiments/edit_direction_alignment/scripts/haufe_edit.py::othello` (PI-haufe: Δz = α Pᵀ δy;
ND-haufe: direction = P[target] − P[current]; canonical Edit Index + fidelity guard). Nothing
canonical is touched; output → experiments/adjacent_flip_ablation/scores/haufe_edit_adjacent_flip.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/edit_direction_alignment/scripts"))
from haufe_edit import othello  # noqa: E402

RUN, INST = "runs/adjacent_flip_ablation/L-oth-adjacent-flip-20m", "oth-adjacent-flip"
print(f"othello Haufe-corrected editing on {RUN} (canonical: PI +0.168/0.85, ND +0.237/0.62, unedited -0.697):", flush=True)
res = othello(RUN, INST)
res["run"] = RUN; res["instance"] = INST
out = REPO / "experiments/adjacent_flip_ablation/scores/haufe_edit_adjacent_flip.json"
out.write_text(json.dumps(res, indent=1, default=float)); print("wrote", out.relative_to(REPO), flush=True)
