"""Print the inverse-probe summary table from every `*_mirror128.json` in scores/, one row per
run: unedited, best arm of the four write forms (Edit Index / guard, over residual points),
the run's canonical editors, and the range of g's held-out R² over points 1–8."""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ORDER = ["L-oth-20m", "L-oth-adjacent-flip-20m", "L-oth-adjacent-20m", "L-oth-noflip-20m",
         "L-dw-20m", "L-dw-noiseless-20m", "L-dw-smooth-20m", "L-dw-8ray-20m", "L-dw-8ray-tok-20m", "L-dw-5ray-20m", "L-dw-blink-20m"]
FORMS = (("overwrite", "overwrite"), ("delta", "delta@"), ("nn_overwrite", "nn_overwrite"), ("nn_delta", "nn_delta@"))

def best(P, key):
    c = [a for b in P.values() for k, a in b["arms"].items() if (k == key if not key.endswith("@") else k.startswith(key))]
    return max(c, key=lambda a: a["edit_index"])

def find_run(name):
    hits = [p for p in (REPO / "runs").glob(f"*/{name}/scores.json") if not p.parts[-3].startswith("_")]
    return hits[0] if hits else None

rows = []
for f in (REPO / "experiments/inverse_probe/scores").glob("*_mirror128.json"):
    d = json.load(open(f)); name = d["run"].split("/")[-1]
    sp = find_run(name); s = json.load(open(sp)) if sp else {}
    env = "othello" if f.name.startswith("othello") else "discworld"
    if env == "othello":
        can = {e: (s["best"][e]["edit_index_symdiff"], s["best"][e]["fidelity_ratio"]) for e in ("PI", "ND", "GS")}
        uned = s["unedited"]["edit_index_symdiff"]
    else:
        B = s["bases"].get("frustum") or s["bases"].get("cartesian")
        can = {e: ((B["best"][e]["edit_index"], B["best"][e]["fidelity_ratio"]) if B["best"].get(e) else None) for e in ("PI", "ND", "GS")}
        can["ND"] = None                    # not reported on a discworld regression target
        uned = B["unedited"]["edit_index"]
    P = d["points"]; r2 = [P[str(p)]["g_r2_heldout"] for p in range(1, 9) if str(p) in P]
    cells = []
    for _, key in FORMS:
        a = best(P, key); cells.append(f"{a['edit_index']:+.2f} / {a['fidelity_ratio']:.2f}")
    fmt = lambda v: "n/a" if v is None else f"{v[0]:+.2f} / {v[1]:.2f}"  # noqa: E731
    tag = " †" if s.get("ei_construction") == "frame-set" else ""
    rows.append((ORDER.index(name) if name in ORDER else 99, f"| {name}{tag} | {uned:+.2f} | " + " | ".join(cells)
                 + f" | {fmt(can['PI'])} | {fmt(can['ND'])} | {fmt(can['GS'])} | {min(r2):.2f}–{max(r2):.2f} |"))
print("| run | unedited | overwrite | delta | retrieval overwrite | retrieval delta | canonical PI | canonical ND | canonical GS | state explains h (R², pts 1–8) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for _, r in sorted(rows):
    print(r)
