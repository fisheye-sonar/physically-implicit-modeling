"""IM verification table: per run and block, best IM / IM-NN vs canonical PI / GS, and — where the
2026-09-14 inverse-probe experiment scored the same run — its mirrored-map overwrite for the wiring check."""
import glob, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
exp = {}
for p in glob.glob(str(REPO / "experiments/inverse_probe/scores/*_mirror128.json")):
    d = json.load(open(p)); ow = max((P["arms"]["overwrite"] for P in d["points"].values()), key=lambda a: a["edit_index"])
    exp[d["run"].split("/")[-1]] = (ow["edit_index"], ow["fidelity_ratio"])
def cell(a, k="edit_index"):
    return "—" if not a else f"{a[k]:+.2f}/{a['fidelity_ratio']:.2f}"
rows, n_im, n_blocks = [], 0, 0
for p in sorted(glob.glob(str(REPO / "runs/*/*/scores.json"))):
    top = p.split("/")[-3]
    if top.startswith("_") or top == "archive": continue
    d = json.load(open(p)); run = p.split("/")[-2]
    blocks = {**({"mine/theirs": d} if "arms" in d else {}), **d.get("bases", {})}
    for key, blk in blocks.items():
        n_blocks += 1; b = blk.get("best", {}); has = any(a.get("editor") == "IM" for a in blk.get("arms", []))
        n_im += has
        k = "edit_index_symdiff" if key == "mine/theirs" else "edit_index"
        e = exp.get(run) if key in ("frustum", "cartesian", "mine/theirs") else None
        rows.append(f"{top}/{run:34s} {key:16s} IM {cell(b.get('IM'), k):12s} IM-NN {cell(b.get('IM-NN'), k):12s} "
                    f"PI {cell(b.get('PI'), k):12s} GS {cell(b.get('GS'), k):12s}" + (f" | experiment overwrite {e[0]:+.2f}/{e[1]:.2f}" if e else ""))
print("\n".join(rows)); print(f"\nblocks with IM: {n_im}/{n_blocks}")
