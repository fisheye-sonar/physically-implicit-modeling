"""One line per new probe-target block: skill and the best arm of each editor (for the pings)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = ["initial_othello_comparison/L-oth-20m", "objective_ablation/L-oth-20m-mse",
        "flip_ablation/L-oth-noflip-20m", "adjacency_ablation/L-oth-adjacent-20m",
        "ray_ablation/L-dw-8ray-20m", "interface_ablation/L-dw-8ray-tok-20m"]
SKIP = {"cartesian", "frustum"}
for r in RUNS:
    sp = REPO / "runs" / r / "scores.json"
    if not sp.exists():
        continue
    s = json.loads(sp.read_text())
    for key, T in s.get("bases", {}).items():
        if key in SKIP:
            continue
        best = T.get("best", {})
        arms = " ".join(f"{ed} {b['edit_index']:+.3f}/{b['fidelity_ratio']:.2f}"
                        for ed, b in best.items() if b)
        print(f"{r.split('/')[1]:22s} {key:16s} skill LIN {max(T['probe_skill_linear']):+.3f} "
              f"MLP {max(T['probe_skill_mlp']):+.3f} | unedited {T['unedited']['edit_index']:+.3f} | {arms}")
