#!/usr/bin/env python
"""Table 3's last two columns — PI through the HAUFE-corrected directions, at each row's best PI
point, on the canonical bench (2026-09-12; queued behind the rescoring).

    .pim/bin/python experiments/edit_direction_alignment/scripts/table3_haufe_edit.py [--only <substr>]

For every Table 3 row (see ``table3_alignment.py``): the run's canonical linear probe at the
best PI point of the CURRENT scores.json, the Haufe pattern matrix P = [Σ Wᵀ (W Σ Wᵀ)⁻¹]ᵀ from the
z-space residual covariance at that point (Σ over 2000 probe-corpus sequences / the last 2000
probe games), and PI-haufe: Δz = α · Pᵀ (target read-out − current read-out) with the SAME
target PI uses (regression: the bench target, the rest held; categorical: the probe's own
read-out with the class swaps — ``arms.pinv_target``), swept over the shared α grid, dims
"all". Scored with the canonical construction on the canonical 1000-case bench: discworld
``arms.score`` (ray-zone / frame-set), Othello the symmetric-difference index with the guard.
The reported cell is the max-Edit-Index α (Table 2's rule).

Writes ``scores/table3_haufe_edit.json``: [{run, env, target, point, alpha, pi_ei, pi_fid,
unedited, n_cases, note}]. Nothing canonical changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import haufe_patterns  # noqa: E402

from pim.environments import layout  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import K_ROLL  # noqa: E402
from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.data import N_CLASSES, board_probs  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = _REPO / "experiments" / "edit_direction_alignment" / "scores" / "table3_haufe_edit.json"
ALPHA_CAT_PI = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 35.0, 60.0, 100.0)
ALPHA_REG_PI = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0)
N_BENCH = 1000
ROWS = [
    ("initial_othello_comparison/L-oth-20m", "othello", "mine/theirs"),
    ("objective_ablation/L-oth-20m-mse", "othello", "mine/theirs"),
    ("flip_ablation/L-oth-noflip-20m", "othello", "mine/theirs"),
    ("adjacency_ablation/L-oth-adjacent-20m", "othello", "mine/theirs"),
    ("adjacent_flip_ablation/L-oth-adjacent-flip-20m", "othello", "mine/theirs"),
    ("initial_othello_comparison/L-dw-20m", "discworld", "frustum"),
    ("noise_ablation/L-dw-noiseless-20m", "discworld", "frustum"),
    ("noise_ablation/L-dw-noiseless-20m", "discworld", "appearance-fac"),
    ("ray_ablation/L-dw-8ray-20m", "discworld", "frustum"),
    ("ray_ablation/L-dw-8ray-20m", "discworld", "appearance-fac"),
    ("interface_ablation/L-dw-8ray-tok-20m", "discworld", "frustum"),
    ("interface_ablation/L-dw-8ray-tok-20m", "discworld", "appearance-fac"),
    ("ray_ablation/L-dw-5ray-20m", "discworld", "frustum"),
    ("ray_ablation/L-dw-5ray-20m", "discworld", "appearance-fac"),
    ("blink_ablation/L-dw-blink-20m", "discworld", "frustum"),
    ("blink_ablation/L-dw-blink-20m", "discworld", "appearance-fac"),
]


def log(m):
    print(m, flush=True)


def best_pi_point(scores: dict, env: str, target: str) -> int | None:
    b = scores["best"] if env == "othello" else scores.get("bases", {}).get(target, {}).get("best", {})
    return int(b["PI"]["point"]) if b and b.get("PI") else None


def patterns(probe, z_all: torch.Tensor) -> torch.Tensor:
    return haufe_patterns(probe.net.weight.detach(), torch.cov(z_all.T))


# ── discworld (frames and tokens) ────────────────────────────────────────────


def discworld_row(run_rel: str, target: str) -> dict | None:
    run_dir = _REPO / "runs" / run_rel
    s = json.loads((run_dir / "scores.json").read_text())
    ell = best_pi_point(s, "discworld", target)
    if ell is None:
        return None
    inst = s["instance"]
    tokens = s["arch"].endswith("_tokens")
    tgt_name = "full" if target == "frustum" else target
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    recipe = dwa.probe_recipe(tgt_name, inst, n_seq=30_000)
    enc = {}
    if tokens:
        from pim.environments.discworld import token_bench as tkb
        from pim.environments.discworld.tokens import FrameVocab
        vocab = FrameVocab.load(run_dir / "vocab.npz")
        _e, _tag = tkb.token_encoder(vocab)
        enc = {"encoder": _e, "encoder_tag": _tag}
    lin = dwa.fit_probes(model, target=tgt_name, family="linear", basis_name="frustum", cache_dir=run_dir / "probes",
                         log=None, require_cached=True, **recipe, **enc)
    probe = lin[ell][0]
    # Σ at the point over 2000 held-out probe-corpus sequences (outside the fit rows)
    with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
        obs_cov = f["obs_intensity"][30_000:32_000, : int(getattr(model, "state_span", 39))].astype(np.float32)
    if tokens:
        obs_cov = enc["encoder"](obs_cov)
    R = collect_residuals(model, obs_cov, batch=64, points=[ell])
    z_all = (torch.from_numpy(np.asarray(R[0]).reshape(-1, R.shape[-1])).to(DEV) - probe.x_mean) / probe.x_std
    P = patterns(probe, z_all)                                          # (d_out, d)
    del R
    alphas = ALPHA_REG_PI if target == "frustum" else ALPHA_CAT_PI
    arms = []
    if not tokens:
        b = dwb.load_bench(model, n=N_BENCH, target=tgt_name, basis_name="frustum", instance=inst)
        u = dwa.unsteered(model, b)
        dwa.as_activations(model, ell)
        h0 = model.flat_state(b.state)
        z0 = (h0 - probe.x_mean) / probe.x_std
        with torch.no_grad():
            if b.kind == "regression":
                y_star = torch.where(b.change_mask, b.tgt, probe(h0))    # the write target, the rest held
                d_y = (y_star - probe.y_mean) / probe.y_std - probe.net(z0)
            else:
                d_y = dwa.pinv_target(probe, h0, b) - probe.net(z0).reshape(h0.shape[0], -1)
            dz = d_y @ P
        for a in alphas:
            with torch.no_grad():
                roll = model.rollout_with_edit(b.state, ell, h0 + a * dz * probe.x_std, K_ROLL).cpu().numpy()
            c = dwa.score(model, b, roll, u)
            arms.append({"alpha": a, "pi_ei": c["edit_index"], "pi_fid": c["fidelity_ratio"]})
        unedited, n = u["edit_index"], b.n
    else:
        tb = tkb.load_token_bench(vocab, n=N_BENCH, target=tgt_name, basis_name="frustum", instance=inst)
        uns, u = tkb.unsteered(model, tb)
        h0 = tkb.residuals_last(model, tb)[ell]
        z0 = (h0 - probe.x_mean) / probe.x_std
        with torch.no_grad():
            if tb.kind == "regression":
                y_star = torch.where(tb.change_mask, tb.tgt, probe(h0))
                d_y = (y_star - probe.y_mean) / probe.y_std - probe.net(z0)
            else:
                d_y = dwa.pinv_target(probe, h0, tb) - probe.net(z0).reshape(h0.shape[0], -1)
            dz = d_y @ P
        for a in alphas:
            probs = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(ell, h0 + a * dz * probe.x_std))
            c = tkb.scorecard(probs, tb, uns)
            arms.append({"alpha": a, "pi_ei": c["edit_index"], "pi_fid": c["fidelity_ratio"]})
        unedited, n = u["edit_index"], tb.n
    best = max(arms, key=lambda r: r["pi_ei"])
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"run": run_rel, "env": "discworld", "target": target, "point": ell, **best, "unedited": unedited,
            "n_cases": int(n), "arms": arms, "note": "PI-haufe, dims all, Σ over probe seqs 30000-31999 at the point"}


# ── othello ──────────────────────────────────────────────────────────────────


def othello_row(run_rel: str) -> dict | None:
    run_dir = _REPO / "runs" / run_rel
    s = json.loads((run_dir / "scores.json").read_text())
    ell = best_pi_point(s, "othello", "mine/theirs")
    if ell is None:
        return None
    inst = s["instance"]
    rules = oc.rules_of(inst)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    kind = getattr(model, "output_kind", "logits")
    paths = oc.build(only=("probe",), instance=inst, log=lambda m: None)
    data = oc.probe_data(paths["probe"], 20_000, **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    probe = grid.probes[("mine", "linear", "sequence", ell)]
    with torch.no_grad():
        rs = model.residual_stack(torch.from_numpy(data.tokens[18_000:20_000]).to(DEV))
        m = torch.from_numpy(data.mask[18_000:20_000]).to(DEV)
        z_all = (rs[ell][m] - probe.x_mean) / probe.x_std
    P = patterns(probe, z_all)
    del rs
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    uns = oa.unsteered_probs(model, bench)
    u = move_scorecard(uns, bench.legal_pre, bench.legal_post)
    arms = []
    for a in ALPHA_CAT_PI:
        probs = np.zeros((bench.n_cases, 64), np.float32)
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx = torch.from_numpy(toks).to(DEV)
            bsz = len(ids)
            sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
            td = torch.from_numpy(tgt[ids]).to(DEV)
            cd = torch.from_numpy(cur[ids]).to(DEV)

            def hook(layer, x, _a=a, _sq=sq, _td=td, _cd=cd, _bsz=bsz):
                if layer != ell:
                    return x
                c = x[:, -1]
                z = (c - probe.x_mean) / probe.x_std
                ar = torch.arange(_bsz, device=c.device)
                lg = probe.net(z).view(_bsz, -1, N_CLASSES).clone()
                sel = lg[ar, _sq].clone()
                new = sel.clone()
                new[ar, _td] = sel[ar, _cd]
                new[ar, _cd] = sel[ar, _td]
                lg[ar, _sq] = new
                d_y = lg.view(_bsz, -1) - probe.net(z)
                o = x.clone()
                o[:, -1] = c + _a * (d_y @ P) * probe.x_std
                return o
            with torch.no_grad():
                probs[ids] = board_probs(model.decode(idx, edit=hook), kind)
        card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
        arms.append({"alpha": a, "pi_ei": card["edit_index_symdiff"],
                     "pi_fid": move_fidelity_ratio(probs, uns, bench.legal_post)})
    best = max(arms, key=lambda r: r["pi_ei"])
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"run": run_rel, "env": "othello", "target": "mine/theirs", "point": ell, **best,
            "unedited": u["edit_index_symdiff"], "n_cases": int(bench.n_cases), "arms": arms,
            "note": "PI-haufe on the single-flip bench; symmetric-difference index; Σ over probe games 18000-19999"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="substrings of run names")
    a = ap.parse_args()
    prev = {(r["run"], r["target"]): r for r in json.loads(OUT.read_text())} if OUT.exists() else {}
    for run_rel, env, target in ROWS:
        if a.only and not any(o in run_rel for o in a.only):
            continue
        if not (_REPO / "runs" / run_rel / "scores.json").exists():
            log(f"skip {run_rel}: no scores.json")
            continue
        t0 = time.time()
        try:
            r = othello_row(run_rel) if env == "othello" else discworld_row(run_rel, target)
        except RuntimeError as e:                       # e.g. probes for a target not fitted
            log(f"skip {run_rel} / {target}: {str(e).splitlines()[0][:120]}")
            continue
        if r is None:
            log(f"skip {run_rel} / {target}: no PI arm in scores.json")
            continue
        r["minutes"] = round((time.time() - t0) / 60, 2)
        prev[(r["run"], r["target"])] = r
        log(f"{run_rel:<48} {target:<15} pt {r['point']}  PI-haufe {r['pi_ei']:+.3f} / fid {r['pi_fid']:.2f} "
            f"(α {r['alpha']:g})  unedited {r['unedited']:+.3f}  [{r['minutes']} min]")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(list(prev.values()), indent=1, default=float))
    log(f"-> {OUT.relative_to(_REPO)}  ({len(prev)} rows)")


if __name__ == "__main__":
    main()
