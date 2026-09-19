"""Probe-corpus-size control, discworld (2026-09-19, Sevan): do decodability and EDITABILITY move
when the regression probes and the inverse map are fitted on more sequences than the canonical
30k? Everything is the canonical pipeline with ``n_seq`` as the only knob.

    python -u experiments/probe_corpus_size/scripts/corpus_size_dw.py \
        --run noise_ablation/L-dw-noiseless-20m --basis cartesian --sizes 30000 100000 200000

For each size n (the first n sequences of the instance's LARGE probe split, ``probe_250k`` — one
corpus for every size, so the curve is within-corpus; the run's canonical numbers, fitted on 30k of
``probe_120k``, are recorded beside it as the reference): fit LIN and MLP-128 on the FULL state in
``--basis`` with the canonical recipe (``arms.fit_probes``, 80/20 by sequence, seed 0; cached in
this experiment's ``probes/``), sweep PI and GS on the run's canonical bench with the run's own
alpha grids / GS layers, and — for n ≤ ``--im-max-n`` — fit the inverse map and write IM / IM-NN
(``arms.inverse_arms``). No metric is computed here: skill is ``probe_skill_from_stats``, every
arm record is the canonical scorecard with the guard attached. Results are written after EVERY
size (``scores/dw_<run>_<basis>.json``), so a killed job keeps what it finished.

Memory: the dense regression path memmaps the residual stack on disk (9 points × n × 39 × 512 × 4 B:
144 GB at 200k, in ``.scratch/``) and loads one point's training rows at a time (12.8 GB at 200k).
The inverse map holds a point's residuals AND a retrieval bank on the GPU — ~25 GB of GPU and ~32 GB
of RAM at 200k, under a 45 GB unit cap whose kill cannot be caught — hence ``--im-max-n 100000``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments" / "probe_corpus_size"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--basis", default="cartesian")
ap.add_argument("--sizes", nargs="+", type=int, default=(30_000, 100_000, 200_000))
ap.add_argument("--corpus", default="250k", help="the instance's probe split the sizes are cut from")
ap.add_argument("--im-max-n", type=int, default=100_000, help="largest n the inverse map is fitted at")
ap.add_argument("--smoke", action="store_true", help="2k sequences, 2 alphas, 1 GS layer, scratch cache and output")
a = ap.parse_args()

run = REPO / "runs" / a.run
scores = json.loads((run / "scores.json").read_text())
S, inst, B = scores["settings"], json.loads((run / "config.json").read_text())["data"]["instance"], a.basis
model, _ = load_checkpoint(run / "best_model.pt", device=dwa.DEV)
model.eval()
cache_dir = EXP / "probes" if not a.smoke else REPO / ".scratch" / "probe_corpus_size_smoke"
out_path = (EXP / "scores" / f"dw_{run.name}_{B}.json") if not a.smoke else REPO / ".scratch" / "probe_corpus_size_smoke_dw.json"
sizes = [2000] if a.smoke else list(a.sizes)
a_pi, a_gs, layers = S["dw_alpha_pi"], S["dw_alpha_gs"], S["gs_layers"]
if a.smoke:
    a_pi, a_gs, layers = a_pi[4:6], a_gs[4:6], layers[:1]
dimsets = tuple(S["dw_edit_dims"])


def best(arms, name):
    sub = [r for r in arms if r["editor"] == name or r["editor"].startswith(name + "[") or r["editor"].startswith(name + "@")]
    if not sub:
        return None
    b = max(sub, key=lambda r: r["edit_index"])
    return {k: v for k, v in b.items() if np.isscalar(v)}


canon = scores["bases"].get(B, {})
res = json.loads(out_path.read_text()) if out_path.exists() else {}
res.update({"run": a.run, "instance": inst, "basis": B, "corpus": f"probe_{a.corpus}", "bench_n": S["dw_bench_n"],
            "canonical": {"n_seq": S["dw_probe_seqs"], "corpus": "probe_120k",
                          "skill_LIN": max(canon["probe_skill_linear"]) if canon else None,
                          "skill_MLP": max(canon["probe_skill_mlp"]) if canon else None,
                          "best": {e: (canon.get("best", {}).get(e) and {k: canon["best"][e].get(k) for k in
                                       ("edit_index", "fidelity_ratio", "point", "alpha")}) for e in ("PI", "GS", "IM", "IM-NN")},
                          "g_r2_max": max(canon["inverse_map"]["g_r2"]) if canon.get("inverse_map") else None}})
res.setdefault("sizes", {})
b = dwb.load_bench(model, n=S["dw_bench_n"], target="full", basis_name=B, instance=inst)
u = dwa.unsteered(model, b)
res["unedited"] = u["edit_index"]
t0 = time.time()
for n in sizes:
    recipe = {"probe": {"instance": inst, "size": a.corpus}, "n_seq": int(n), "epochs": None}
    rec = res["sizes"].get(str(n), {})
    lin = dwa.fit_probes(model, target="full", family="linear", basis_name=B, cache_dir=cache_dir, log=None, **recipe)
    mlp = dwa.fit_probes(model, target="full", family="mlp", basis_name=B, cache_dir=cache_dir, log=None, **recipe)
    rec["rows_train"] = int(0.8 * n) * 39
    rec["skill_LIN_by_point"] = [probe_skill_from_stats(lin[e][1]) for e in sorted(lin)]
    rec["skill_MLP_by_point"] = [probe_skill_from_stats(mlp[e][1]) for e in sorted(mlp)]
    rec["skill_LIN"], rec["skill_MLP"] = max(rec["skill_LIN_by_point"]), max(rec["skill_MLP_by_point"])
    print(f"n={n:,}: skill LIN {rec['skill_LIN']:.4f} MLP {rec['skill_MLP']:.4f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    arms = []
    for dims in dimsets:
        arms += dwa.pinv_arm(model, b, lin, a_pi, space="zspace", dims=dims)
        arms += dwa.grad_steer_arm(model, b, mlp, layers, a_gs, n_steps=S["dw_gs_steps"], beta=S["dw_gs_beta"], dims=dims)
    for r in arms:
        r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)
        r.update(dwa.fidelity_ci95(r, u))
    rec["best"] = {"PI": best(arms, "PI"), "GS": best(arms, "GS")}
    print("   " + " · ".join(f"{k} {v['edit_index']:+.3f}/{v['fidelity_ratio']:.2f} (pt {v['point']}, α {v['alpha']:g})"
                             for k, v in rec["best"].items() if v) + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    res["sizes"][str(n)] = rec
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))
    del lin, mlp
    if n <= a.im_max_n:
        im_arms, st = dwa.inverse_arms(model, {B: b}, basis_name=B, unsteered_cards={B: u}, cache_dir=cache_dir,
                                       log=None, n_seq=int(n), probe=recipe["probe"])
        rec["best"]["IM"], rec["best"]["IM-NN"] = best(im_arms[B], "IM"), best(im_arms[B], "IM-NN")
        rec["g_r2_by_point"], rec["nn_r2_by_point"] = st["g_r2"], st["nn_r2"]
        print("   " + " · ".join(f"{k} {rec['best'][k]['edit_index']:+.3f}/{rec['best'][k]['fidelity_ratio']:.2f} (pt {rec['best'][k]['point']})"
                                 for k in ("IM", "IM-NN")) + f" · g R² max {max(st['g_r2']):+.3f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    else:
        rec["best"]["IM"] = rec["best"]["IM-NN"] = None
        rec["im_skipped"] = f"n > --im-max-n {a.im_max_n} (GPU/RAM: the retrieval bank and the fit hold a point's residuals twice)"
    res["sizes"][str(n)] = rec
    res["written"] = time.strftime("%Y-%m-%d %H:%M")
    out_path.write_text(json.dumps(res, indent=1, default=float))
print("wrote", out_path.relative_to(REPO), f"[{(time.time() - t0) / 60:.1f} min]")
