"""The INLP sweep summary table — one row per run, extended as runs land (EI / guard everywhere)."""
import json
from pathlib import Path
SC = Path(__file__).resolve().parents[1] / "scores"
ORDER = [("L-dw-noiseless-20m", ""), ("L-dw-8ray-20m", ""), ("L-dw-smooth-20m", ""), ("L-dw-8ray-tok-20m", ""), ("L-dw-noiseless-20m", "_fac"), ("L-dw-8ray-20m", "_fac"), ("L-dw-5ray-20m", ""), ("L-dw-5ray-20m", "_fac")]
def cell(r): return "—" if r is None else f"{r['edit_index']:+.2f} / {r['fidelity_ratio']:.2f}"
def lst(d, key, fmt=lambda x: str(x)):
    """per point 1–8: the value; '∞' = never reached within the cascade; '…' = point not computed yet"""
    return " / ".join("…" if str(p) not in d["points"] else ("∞" if d["points"][str(p)].get(key) is None else fmt(d["points"][str(p)][key])) for p in range(1, 9))
print("| run (target) | unedited | dimensions removed per variable to mean R² < 0.4, pts 1–8 | … to mean R² < 0.05, pts 1–8 | … to exhaustion (R² < 0.02), pts 1–8 | best guarded K-copy write, shrink (pt, K, α) | K = 1 at that point (≡ PI, lstsq) | best unguarded K-copy write | canonical PI | canonical ND | canonical GS |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name, suf in ORDER:
    p = SC / f"inlp_{name}{suf}.json"; label = name + (" (appearance-fac)" if suf else " (full state)")
    if not p.exists():
        print(f"| {label} | not yet |"); continue
    d = json.load(open(p)); P = d["points"]; pts = [k for k in P if "best_shrink" in P[k]]
    if not pts:
        print(f"| {label} | in progress ({len(P)} points) |"); continue
    bu_p = max(pts, key=lambda k: P[k]["best_shrink"]["edit_index"]); bu = P[bu_p]["best_shrink"]
    g = [(k, P[k]["best_shrink_guarded"]) for k in pts if P[k].get("best_shrink_guarded")]
    bg = max(g, key=lambda kv: kv[1]["edit_index"]) if g else None
    hp = bg[0] if bg else bu_p                                    # the headline point: best guarded, else best unguarded
    k1 = P[hp]["k1_exact_best"]; c = d["canonical"]; tok = " †" if d.get("model_kind") == "tokens" else ""
    print(f"| {label}{tok} | {d['unedited']['edit_index']:+.2f} | {lst(d, 'iters_to_0.4')} | {lst(d, 'iters_to_0.05')} | {lst(d, 'k_exhaust_mean', lambda x: f'{x:.0f}')} | "
          f"{'—' if bg is None else cell(bg[1]) + f' (pt {bg[0]}, K {bg[1][chr(75)]}, α {bg[1][chr(97)+chr(108)+chr(112)+chr(104)+chr(97)]:g})'} | {cell(k1)} | "
          f"{cell(bu)} (pt {bu_p}, K {bu['K']}{' = all' if bu.get('K_is_all') else ''}) | {cell(c.get('PI'))} | {cell(c.get('ND'))} | {cell(c.get('GS'))} |")
