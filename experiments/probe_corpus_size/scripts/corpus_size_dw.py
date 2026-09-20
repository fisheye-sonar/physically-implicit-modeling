"""Probe-corpus-size control, discworld (2026-09-19, Sevan): do decodability and EDITABILITY move
when the regression probes and the inverse map are fitted on more sequences than the canonical
30k? Everything is the canonical pipeline with ``n_seq`` as the only knob.

    python -u experiments/probe_corpus_size/scripts/corpus_size_dw.py \
        --run noise_ablation/L-dw-noiseless-20m --basis cartesian --sizes 30000 100000 --im-sizes 30000 60000

For each size n (the first n sequences of the instance's LARGE probe split, ``probe_250k`` — one
corpus for every size, so the curve is within-corpus; the run's canonical numbers, fitted on 30k of
``probe_120k``, are recorded beside it as the reference): for n in ``--sizes`` fit LIN and MLP-128
on the FULL state in ``--basis`` with the canonical recipe (``arms.fit_probes``, 80/20 by sequence,
seed 0; cached in this experiment's ``probes/``) and sweep PI and GS on the run's canonical bench
with the run's own alpha grids / GS layers; for n in ``--im-sizes`` fit the inverse map and write
IM / IM-NN (``arms.inverse_arms``). No metric is computed here: skill is
``probe_skill_from_stats``, every arm record is the canonical scorecard with the guard attached.
Results are written after every part of every size (``scores/dw_<run>_<basis>.json``) and a part
already recorded there is SKIPPED — a retry, or a second job adding a size, costs nothing.

MEMORY. The dense regression path memmaps the residual stack on disk (9 points × n × 39 × 512 × 4 B:
72 GB at 100k, 144 GB at 200k, in ``.scratch/``) and loads one point's training rows at a time
(12.8 GB at 200k, before the fit's own copies). The inverse map holds a point's rows about five times
over — the Othello sibling was OOM-killed at 40.7 GB with 3.5M rows (2026-09-20) — so IM runs at
30k / 60k sequences (1.2M / 2.3M rows), and the 200k probe size runs as its OWN queue job with one
attempt, so that an out-of-memory kill there cannot take the smaller sizes with it.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import arms as dwa, bench as dwb  # noqa: E402
from pim.metrics.decodability import probe_skill_from_stats  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments" / "probe_corpus_size"
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--basis", default="cartesian")
ap.add_argument("--sizes", nargs="*", type=int, default=(30_000, 100_000),
                help="sequences LIN + MLP-128 and the PI / GS sweeps are fitted at")
ap.add_argument("--im-sizes", nargs="*", type=int, default=(30_000, 60_000),
                help="sequences the inverse map + IM / IM-NN are fitted at")
ap.add_argument("--corpus", default="250k", help="the instance's probe split the sizes are cut from")
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
im_sizes = [2000] if a.smoke else list(a.im_sizes)
a_pi, a_gs, layers = S["dw_alpha_pi"], S["dw_alpha_gs"], S["gs_layers"]
if a.smoke:
    a_pi, a_gs, layers = a_pi[4:6], a_gs[4:6], layers[:1]
dimsets = tuple(S["dw_edit_dims"])
b = dwb.load_bench(model, n=S["dw_bench_n"], target="full", basis_name=B, instance=inst)
u = dwa.unsteered(model, b)
t0 = time.time()


def mins() -> str:
    return f"[{(time.time() - t0) / 60:.1f} min]"


def best(arms, name):
    sub = [r for r in arms if r["editor"] == name or r["editor"].startswith(name + "[") or r["editor"].startswith(name + "@")]
    if not sub:
        return None
    bb = max(sub, key=lambda r: r["edit_index"])
    return {k: v for k, v in bb.items() if np.isscalar(v)}


def save(res: dict) -> None:
    res["written"] = time.strftime("%Y-%m-%d %H:%M")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, default=float))


def probe_part(n: int, recipe: dict, rec: dict) -> None:
    """LIN + MLP-128 at n sequences, then PI and GS swept; best arm per editor."""
    lin = dwa.fit_probes(model, target="full", family="linear", basis_name=B, cache_dir=cache_dir, log=None, **recipe)
    mlp = dwa.fit_probes(model, target="full", family="mlp", basis_name=B, cache_dir=cache_dir, log=None, **recipe)
    rec["rows_train"] = int(0.8 * n) * 39
    rec["skill_LIN_by_point"] = [probe_skill_from_stats(lin[e][1]) for e in sorted(lin)]
    rec["skill_MLP_by_point"] = [probe_skill_from_stats(mlp[e][1]) for e in sorted(mlp)]
    rec["skill_LIN"], rec["skill_MLP"] = max(rec["skill_LIN_by_point"]), max(rec["skill_MLP_by_point"])
    print(f"n={n:,}: skill LIN {rec['skill_LIN']:.4f} MLP {rec['skill_MLP']:.4f}  {mins()}", flush=True)
    arms = []
    for dims in dimsets:
        arms += dwa.pinv_arm(model, b, lin, a_pi, space="zspace", dims=dims)
        arms += dwa.grad_steer_arm(model, b, mlp, layers, a_gs, n_steps=S["dw_gs_steps"], beta=S["dw_gs_beta"], dims=dims)
    for r in arms:
        r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)
        r.update(dwa.fidelity_ci95(r, u))
    rec["best"]["PI"], rec["best"]["GS"] = best(arms, "PI"), best(arms, "GS")
    print("   " + " · ".join(f"{k} {rec['best'][k]['edit_index']:+.3f}/{rec['best'][k]['fidelity_ratio']:.2f} "
                             f"(pt {rec['best'][k]['point']}, α {rec['best'][k]['alpha']:g})" for k in ("PI", "GS") if rec["best"][k])
          + f"  {mins()}", flush=True)


def inverse_part(n: int, recipe: dict, rec: dict) -> None:
    """The inverse map at n sequences and its IM / IM-NN arms; best arm per editor."""
    im_arms, st = dwa.inverse_arms(model, {B: b}, basis_name=B, unsteered_cards={B: u}, cache_dir=cache_dir,
                                   log=None, n_seq=int(n), probe=recipe["probe"])
    rec["best"]["IM"], rec["best"]["IM-NN"] = best(im_arms[B], "IM"), best(im_arms[B], "IM-NN")
    rec["g_r2_by_point"], rec["nn_r2_by_point"] = st["g_r2"], st["nn_r2"]
    print("   " + " · ".join(f"{k} {rec['best'][k]['edit_index']:+.3f}/{rec['best'][k]['fidelity_ratio']:.2f} "
                             f"(pt {rec['best'][k]['point']})" for k in ("IM", "IM-NN"))
          + f" · g R² max {max(st['g_r2']):+.3f}  {mins()}", flush=True)


canon = scores["bases"].get(B, {})
res = json.loads(out_path.read_text()) if out_path.exists() else {}
res.update({"run": a.run, "instance": inst, "basis": B, "corpus": f"probe_{a.corpus}", "bench_n": S["dw_bench_n"],
            "unedited": u["edit_index"],
            "canonical": {"n_seq": S["dw_probe_seqs"], "corpus": "probe_120k",
                          "skill_LIN": max(canon["probe_skill_linear"]) if canon else None,
                          "skill_MLP": max(canon["probe_skill_mlp"]) if canon else None,
                          "best": {e: (canon.get("best", {}).get(e) and {k: canon["best"][e].get(k) for k in
                                       ("edit_index", "fidelity_ratio", "point", "alpha")}) for e in ("PI", "GS", "IM", "IM-NN")},
                          "g_r2_max": max(canon["inverse_map"]["g_r2"]) if canon.get("inverse_map") else None}})
res.setdefault("sizes", {})
for n in sorted(set(sizes) | set(im_sizes)):
    rec = res["sizes"].get(str(n), {})
    rec.setdefault("best", {})
    need_probes = n in sizes and not (rec.get("skill_MLP") is not None and {"PI", "GS"} <= set(rec["best"]))
    need_im = n in im_sizes and not rec["best"].get("IM")
    if not (need_probes or need_im):
        print(f"n={n:,}: already recorded — skipped", flush=True)
        continue
    recipe = {"probe": {"instance": inst, "size": a.corpus}, "n_seq": int(n), "epochs": None}
    if need_probes:
        probe_part(n, recipe, rec)
        res["sizes"][str(n)] = rec
        save(res)
        gc.collect()
        torch.cuda.empty_cache()
    if need_im:
        inverse_part(n, recipe, rec)
        res["sizes"][str(n)] = rec
        save(res)
    gc.collect()
    torch.cuda.empty_cache()
save(res)
print("wrote", out_path.relative_to(REPO), mins())
