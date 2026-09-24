"""dw-blink by subset, in the standard inverse-probe table format (EI / guard, best over points)."""
import json
from pathlib import Path
SC = Path(__file__).resolve().parents[1] / "scores"
def best(arms, prefix):
    ks = [k for k in arms if k == prefix or k.startswith(prefix + "@")]
    return max((arms[k] for k in ks), key=lambda r: r["edit_index"]) if ks else None
def cell(r): return "—" if r is None else f"{r['edit_index']:+.2f} / {r['fidelity_ratio']:.2f}"
rows = [("all cases (canonical bench)", SC / "discworld_L-dw-blink-20m_mirror128.json"),
        ("reappearance (blink ends at the last context frame)", SC / "discworld_L-dw-blink-20m_mirror128_sel-reappearance.json"),
        ("visible (no blink on either object through the rollout)", SC / "discworld_L-dw-blink-20m_mirror128_sel-visible.json")]
print("| L-dw-blink-20m subset | n | unedited | overwrite | delta | retrieval overwrite | retrieval delta | canonical PI | canonical GS | R² (pts 1–8) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, p in rows:
    if not p.exists():
        print(f"| {name} | not yet |"); continue
    d = json.load(open(p)); P = d["points"]
    over = {k: max((P[pt]["arms"][k] if k in P[pt]["arms"] else best(P[pt]["arms"], k) for pt in P), key=lambda r: r["edit_index"]) for k in ("overwrite", "nn_overwrite")}
    dl = {k: max((best(P[pt]["arms"], k) for pt in P), key=lambda r: r["edit_index"]) for k in ("delta", "nn_delta")}
    r2 = [P[pt]["g_r2_heldout"] for pt in sorted(P, key=int) if 1 <= int(pt) <= 8]
    c = d.get("canonical_on_cases") or {}
    pi = c["PI"]["best"] if "PI" in c else d["canonical"]["PI"]; gs = c["GS"]["best"] if "GS" in c else d["canonical"]["GS"]
    n = d.get("n_cases", "1000")
    print(f"| {name} | {n} | {d['unedited']['edit_index']:+.2f} | {cell(over['overwrite'])} | {cell(dl['delta'])} | {cell(over['nn_overwrite'])} | {cell(dl['nn_delta'])} | {cell(pi)} | {cell(gs)} | {min(r2):.2f}–{max(r2):.2f} |")
