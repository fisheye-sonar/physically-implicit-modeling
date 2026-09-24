"""Summary of Sevan's last-tile test and the reconstruction control (2026-09-14 evening).
Reads scores/othello_<run>_mirror128_lasttile.json and scores/othello_<run>_mirror128_recon.json."""
import json, sys
from pathlib import Path
SC = Path(__file__).resolve().parents[1] / "scores"
RUNS = [("L-oth-adjacent-20m", "adjacent"), ("L-oth-20m", "standard")]
def best(arms, prefix):
    ks = [k for k in arms if k == prefix or k.startswith(prefix + "@")]
    return max((arms[k] for k in ks), key=lambda r: r["edit_index"]) if ks else None
def fmt(r): return "—" if r is None else f"{r['edit_index']:+.2f} / {r['fidelity_ratio']:.2f}" + (f" (land {r['landed_tile']:.2f})" if "landed_tile" in r else "")
for run, name in RUNS:
    for kind in ("lasttile", "recon"):
        p = SC / f"othello_{run}_mirror128_{kind}.json"
        if not p.exists():
            print(f"## {name} {kind}: not yet"); continue
        d = json.load(open(p))
        if kind == "lasttile":
            print(f"## {name} — LAST-TILE cases ({d['n_cases']}; unedited {d['unedited']['edit_index_symdiff']:+.2f})")
            c = d.get("canonical_on_cases") or {}
            for e in ("ND", "PI", "GS"):
                if e in c:
                    b, g = c[e]["best"], c[e]["best_guarded"]
                    print(f"  canonical {e} on these cases: best {b['edit_index']:+.2f} / {b['fidelity_ratio']:.2f} (pt {b['point']}, α {b['alpha']}); best guarded {fmt(g)}")
            print("| pt | R² | overwrite | delta | retrieval ow | retrieval delta | recon g(s_pre): pre-fid / drift |")
            print("|---|---|---|---|---|---|---|")
            for pt, P in sorted(d["points"].items(), key=lambda kv: int(kv[0])):
                A = P["arms"]; rc = A["recon_overwrite"]
                print(f"| {pt} | {P['g_r2_heldout']:.2f} | {fmt(A.get('overwrite'))} | {fmt(best(A,'delta'))} | {fmt(A.get('nn_overwrite'))} | {fmt(best(A,'nn_delta'))} | {rc['pre_fidelity']:.2f} / {rc['drift_rmse']:.4f} |")
        else:
            print(f"## {name} — RECONSTRUCTION control on the canonical cases ({d['n_cases']}): overwrite with g(s_pre) / nn(s_pre), no edit")
            print("| pt | R² | g(s_pre): EI / post-fid / pre-fid / drift | nn(s_pre): pre-fid / drift |")
            print("|---|---|---|---|")
            for pt, P in sorted(d["points"].items(), key=lambda kv: int(kv[0])):
                A = P["arms"]; rc, nc = A["recon_overwrite"], A["nn_recon_overwrite"]
                print(f"| {pt} | {P['g_r2_heldout']:.2f} | {rc['edit_index']:+.2f} / {rc['fidelity_ratio']:.2f} / {rc['pre_fidelity']:.2f} / {rc['drift_rmse']:.4f} | {nc['pre_fidelity']:.2f} / {nc['drift_rmse']:.4f} |")
    print()
