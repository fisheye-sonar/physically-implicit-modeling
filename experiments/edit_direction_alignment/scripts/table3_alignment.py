#!/usr/bin/env python
"""Table 3 (2026-09-12): how much of the TRUE counterfactual edit displacement Δ = h_cf − h lies
in the probe's row subspace, at ONE residual point per (run, probe target) — the best PI arm's
point from the run's scores.json — against a generic baseline, before and after the Haufe
correction. Alignment quantities only; no editability is computed here.

    rows_frac      mean_i ‖Q_i Q_iᵀ Δ_i‖² / ‖Δ_i‖²,  Q_i = orthonormal basis of the probe rows the
                   edit changes in case i (common.orth / common.frac_in), Δ in the probe's z-space
    rows_generic   the same with Δ_i replaced by z_{π(i)} − z_i, π a fixed-point-free permutation
                   of the kept cases (default_rng(0)) — a displacement to an unrelated case
    haufe_*        the same two numbers with the rows of the Haufe patterns A = Σ Wᵀ (W Σ Wᵀ)⁻¹
                   (common.haufe_patterns) in place of the weight rows W; Σ = z-space residual
                   covariance at the point over 2000 held-out probe-corpus sequences
    *_ratio        frac / generic

Row subspaces. Regression `full` (frustum basis; 8 outputs o0·x, o0·y, o1·x, o1·y, velocities):
the edited object's two position rows. `appearance-fac`: per changed factor tile of the
edited object, the (tile, old) and (tile, new) rows, row = tile·C + class (`bench.moves`).
Othello mine/theirs: per changed tile, the (tile, cur) and (tile, tgt) rows, row = tile·3 + class.

Counterfactuals. Discworld: `arms.counterfactual_history` (the edited object's trajectory
shifted by its teleport over frames 0..EF−1, noise-matched), kept if in-frustum and
collision-free at every frame; dw-blink renders the same shift with the case's blink schedule
and markers (arms.counterfactual_history has no blink path). Token model: the same frames
encoded through the run's vocabulary, kept if no frame is UNK. Othello: `pilot.make_pairs`
(one of the last 4 moves substituted, remaining moves replayed), kept if the model's legal
mass ≥ 0.98 on both histories.

Reads cached probes only (never fits). Writes scores/table3_alignment.json (the contract),
scores/table3_summary.md, and scores/table3_alignment_cases.npz (per-case fractions).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "edit_index_v2_pilot" / "scripts"))
from common import frac_in, haufe_patterns, orth, zspace  # noqa: E402
from pilot import K_BACK, LEGAL_MASS, POOL_OTH, make_pairs  # noqa: E402

from pim.environments import layout  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "edit_direction_alignment"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
N_COV = 2000                        # sequences / games for the residual covariance
COV_LO_DW = 30_000                  # discworld: sequences 30000..31999 — outside the 30k fit rows
OTH_PROBE_N = 20_000                # the canonical Othello probe corpus size

# (run, env, target) — one Table 3 row each
CONDITIONS = [
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
    ("interface_ablation/L-dw-8ray-tok-20m", "discworld-tokens", "frustum"),
    ("interface_ablation/L-dw-8ray-tok-20m", "discworld-tokens", "appearance-fac"),
    ("ray_ablation/L-dw-5ray-20m", "discworld", "frustum"),
    ("ray_ablation/L-dw-5ray-20m", "discworld", "appearance-fac"),
    ("blink_ablation/L-dw-blink-20m", "discworld", "frustum"),
    ("blink_ablation/L-dw-blink-20m", "discworld", "appearance-fac"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def pi_point(run_dir: Path, env: str, target: str) -> int:
    """The best PI arm's residual point from the run's scores.json (the scorer's own choice)."""
    s = json.loads((run_dir / "scores.json").read_text())
    best = s["best"] if env == "othello" else s["bases"][target]["best"]
    return int(best["PI"]["point"])


def derangement(n: int, rng) -> np.ndarray:
    """A permutation with no fixed point, so every generic partner is a DIFFERENT case."""
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def alignment(dz: torch.Tensor, dgen: torch.Tensor, W: torch.Tensor, A: torch.Tensor,
              rows: list[list[int]]) -> dict:
    """Per-case fractions of Δ (dz) and of the generic displacement (dgen), (n, d) each, inside
    span(W[rows_i]) and span(A[rows_i]); rows_i = the probe rows the edit changes in case i."""
    fr, fg, hr, hg = [], [], [], []
    for i, ri in enumerate(rows):
        Br, Bh = orth(W[ri]), orth(A[ri])
        fr.append(float(frac_in(dz[i:i + 1], Br)))
        fg.append(float(frac_in(dgen[i:i + 1], Br)))
        hr.append(float(frac_in(dz[i:i + 1], Bh)))
        hg.append(float(frac_in(dgen[i:i + 1], Bh)))
    fr, fg, hr, hg = map(np.asarray, (fr, fg, hr, hg))
    return {"rows_frac": float(fr.mean()), "rows_generic": float(fg.mean()),
            "rows_ratio": float(fr.mean() / max(fg.mean(), 1e-12)),
            "haufe_frac": float(hr.mean()), "haufe_generic": float(hg.mean()),
            "haufe_ratio": float(hr.mean() / max(hg.mean(), 1e-12)),
            "n_rows": float(np.mean([len(r) for r in rows])),
            "_per_case": {"rows_frac": fr, "rows_generic": fg, "haufe_frac": hr, "haufe_generic": hg,
                          "n_rows": np.array([len(r) for r in rows])}}


def cov_z(probe, R: np.ndarray) -> torch.Tensor:
    """z-space covariance of residual rows R (m, d) under the probe's standardisation."""
    Z = zspace(probe, torch.from_numpy(R).to(DEV))
    return torch.cov(Z.T)


# ── discworld (frames and tokens) ────────────────────────────────────────────


def cf_frames_blink(b, blink: np.ndarray, seed: int = 0) -> np.ndarray:
    """The counterfactual history on a blink instance: the same shift as
    arms.counterfactual_history, rendered with the case's own visibility schedule and toggle
    markers (discworld_alignment.py's construction), noise-matched."""
    from pim.environments.discworld.bench import EF, N_OBJ
    from pim.environments.discworld.blink import paint_markers
    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import object_constants, sim_config_from

    cfg = sim_config_from(b.sim, N_OBJ)
    cfg = type(cfg)(**{**cfg.__dict__, "obs_noise_std": float(b.sim["obs_noise_std"])})
    rad, refl = object_constants(b.sim, N_OBJ)
    idx, k = np.arange(b.n), b.edit_object.astype(int)
    dt = float(b.sim["dt"])
    delta = b.pos[idx, EF, k] - (b.pos[idx, EF - 1, k] + b.vel[idx, EF - 1, k] * dt)
    hist = b.pos[:, :EF].copy()
    hist[idx, :, k] += delta[:, None, :]
    rng = np.random.default_rng(seed)
    out = np.zeros((b.n, EF, b.obs.shape[-1]), np.float32)
    for i in range(b.n):
        v = blink[i]
        for t in range(EF):
            _, ids, inten = render_frame(hist[i, t], rad, refl, cfg, rng=rng, visible=v[t])
            paint_markers(ids, inten, v[t], v[t + 1] if t + 1 < v.shape[0] else None)
            out[i, t] = inten
    return out


def run_discworld(run_rel: str, env: str, target: str) -> dict:
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld import bench as dwb
    from pim.environments.discworld.bench import EF, N_OBJ
    from pim.environments.discworld.sim import fully_in_frustum
    from pim.metrics.zone_editability import sim_config_from

    run_dir = REPO / "runs" / run_rel
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    tokens = env == "discworld-tokens"
    tgt_name = "full" if target == "frustum" else target
    point = pi_point(run_dir, "discworld", target)
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    enc, enc_kw = None, {}
    if tokens:
        from pim.environments.discworld import token_bench as tkb
        from pim.environments.discworld.tokens import FrameVocab
        vocab = FrameVocab.load(run_dir / "vocab.npz")
        enc, tag = tkb.token_encoder(vocab)
        enc_kw = {"encoder": enc, "encoder_tag": tag}
    # the run's cached probes, exactly as the scorer loads them (cache hit or raise)
    lin = dwa.fit_probes(model, target=tgt_name, family="linear", basis_name="frustum",
                         cache_dir=run_dir / "probes", log=None, require_cached=True,
                         **dwa.probe_recipe(tgt_name, inst), **enc_kw)
    probe = lin[point][0]
    # the canonical bench for this target (model-free arrays; counterfactual_history reads
    # obs/pos/vel/edit_object/sim/n — the pilot's wrapping, which also serves the token model)
    arr = dwb.bench_arrays(n=192, target=tgt_name, basis_name="frustum", instance=inst)
    b = SimpleNamespace(obs=arr["obs"], pos=arr["pos"], vel=arr["vel"], edit_object=arr["edit_object"],
                        sim=arr["sim"], n=arr["n"])
    n = b.n
    ar = np.arange(n)
    k = b.edit_object.astype(int)
    cfg_sim = sim_config_from(b.sim, N_OBJ)
    dt = float(b.sim["dt"])
    delta = b.pos[ar, EF, k] - (b.pos[ar, EF - 1, k] + b.vel[ar, EF - 1, k] * dt)
    cf_pos = b.pos[:, :EF].copy()
    cf_pos[ar, :, k] += delta[:, None, :]
    min_sep = cfg_sim.collision_margin * 2.0 * cfg_sim.radius
    valid = np.array([fully_in_frustum(cf_pos[i], cfg_sim.radius, cfg_sim)
                      and (np.linalg.norm(cf_pos[i, :, 0] - cf_pos[i, :, 1], axis=-1) >= min_sep).all()
                      for i in range(n)])
    counts = {"cases": int(n), "valid_counterfactual": int(valid.sum())}
    blink = arr.get("blink_visible")
    cf = dwa.counterfactual_history(b, noise_matched=True) if blink is None else cf_frames_blink(b, blink)
    obs = b.obs[:, :EF].astype(np.float32)
    notes = []
    if blink is not None:
        notes.append("blink: cf rendered with the case's visibility schedule + markers, noise-matched")
    if tokens:
        from pim.environments.discworld.tokens import UNK
        x, x_cf = enc(obs).astype(np.int64), enc(cf).astype(np.int64)
        in_vocab = (x != UNK).all(1) & (x_cf != UNK).all(1)
        valid &= in_vocab
        counts["in_vocab_both"] = int(valid.sum())
        notes.append(f"token model: {int((~in_vocab).sum())} cases with an UNK frame dropped")
    else:
        x, x_cf = obs, cf.astype(np.float32)
    # the probe rows the edit changes, per case
    if target == "frustum":
        rows_all = [[2 * int(k[i]), 2 * int(k[i]) + 1] for i in range(n)]
        has_rows = np.ones(n, bool)
    else:
        mv = arr["moves"]                                   # {"tile","old","new"} (n, F) int64
        C = probe.n_classes
        rows_all = [[int(t) * C + int(c) for t, o, nw in zip(mv["tile"][i], mv["old"][i], mv["new"][i])
                     if o != nw for c in (o, nw)] for i in range(n)]
        has_rows = np.array([len(r) > 0 for r in rows_all])
        counts["factor_tile_changes"] = int((valid & has_rows).sum())
        if (valid & ~has_rows).any():
            notes.append(f"{int((valid & ~has_rows).sum())} valid cases with no changed factor tile dropped")
    keep = np.where(valid & has_rows)[0]
    counts["kept"] = int(len(keep))
    rows = [rows_all[i] for i in keep]
    # residuals at the last context position, at the point
    R = collect_residuals(model, x[keep], batch=64, points=[point])[0][:, -1]
    Rcf = collect_residuals(model, x_cf[keep], batch=64, points=[point])[0][:, -1]
    # the covariance corpus: held-out probe sequences (outside the 30k fit rows)
    import h5py
    span = int(getattr(model, "state_span", 39))
    with h5py.File(layout.probe_file("discworld", inst, "120k"), "r") as f:
        obs_cov = f["obs_intensity"][COV_LO_DW: COV_LO_DW + N_COV, :span].astype(np.float32)
    x_cov = enc(obs_cov).astype(np.int64) if tokens else obs_cov
    Rc = collect_residuals(model, x_cov, batch=64, points=[point])[0]
    cov = cov_z(probe, Rc.reshape(-1, Rc.shape[-1]))
    del Rc
    z = zspace(probe, torch.from_numpy(R).to(DEV))
    zcf = zspace(probe, torch.from_numpy(Rcf).to(DEV))
    dz = zcf - z
    perm = derangement(len(keep), np.random.default_rng(SEED))
    dgen = z[torch.from_numpy(perm).to(DEV)] - z
    W = probe.net.weight.detach()
    A = haufe_patterns(W, cov)
    res = alignment(dz, dgen, W, A, rows)
    res["_per_case"]["delta_norm_over_h_norm"] = (dz.norm(dim=1) / z.norm(dim=1)).cpu().numpy()
    cf_desc = ("edited object's trajectory shifted by its teleport over frames 0..EF-1 "
               "(arms.counterfactual_history, noise-matched); kept if in-frustum and collision-free")
    if tokens:
        cf_desc += "; encoded through the run's frame vocabulary, kept if no UNK"
    out = {"run": run_rel, "env": env, "instance": inst, "target": target, "point": point,
           "n_cases": int(len(keep)), **{k_: v for k_, v in res.items() if k_ != "_per_case"},
           "counterfactual": cf_desc,
           "notes": "; ".join([f"cases {counts}"] + notes)}
    del model
    torch.cuda.empty_cache()
    return out, res["_per_case"]


# ── othello ──────────────────────────────────────────────────────────────────


def cached_grid(model, data, run_dir: Path):
    """The run's cached mine/theirs probe grid — the same key `oa.fit_probe_grid` builds, but a
    miss RAISES instead of starting an 18-probe fit (the brief: cache hits only, never fit)."""
    from pim.environments.othello import arms as oa
    from pim.probes.base import FIT_BATCH, FIT_EPOCHS, FIT_LR

    store = ProbeCache(run_dir / "probes")
    fname, prov = store.key(model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"],
                            splits=["sequence"], holdout=0.2, epochs=FIT_EPOCHS, batch=FIT_BATCH,
                            lr=FIT_LR, seed=0, n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
                            n_points=model.n_layers + 1)
    if store.load(fname, prov, device="cpu") is None:
        raise RuntimeError(f"no cached probe grid {fname} in {store.dir}")
    return oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)


def run_othello(run_rel: str) -> dict:
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.bench import load_benchmark
    from pim.environments.othello.data import N_CLASSES, board_probs, canonical_vocab

    run_dir = REPO / "runs" / run_rel
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    point = pi_point(run_dir, "othello", "mine/theirs")
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    kind = getattr(model, "output_kind", "logits")
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}
    paths = oc.build(only=("probe",), instance=inst, log=lambda s: None)
    data = oc.probe_data(paths["probe"], OTH_PROBE_N, **rules)
    grid = cached_grid(model, data, run_dir)
    probe = grid.probes[("mine", "linear", "sequence", point)]
    # substitution pairs on the instance's bench histories (pilot.make_pairs)
    bench = load_benchmark(inst)
    hists = [[itos[int(t)] for t in row] for toks in bench.tokens for row in toks]
    rng = np.random.default_rng(SEED)
    pairs, pair_log = make_pairs(hists, rules, K_BACK, POOL_OTH, rng)
    n = len(pairs)
    d = probe.net.weight.shape[1]
    R = torch.zeros(n, d, device=DEV)
    Rcf = torch.zeros(n, d, device=DEV)
    mass_a, mass_b = np.zeros(n), np.zeros(n)
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(pairs):
        by_len.setdefault(len(p["hist_a"]), []).append(i)
    with torch.no_grad():
        for L, ids in sorted(by_len.items()):
            ta = torch.tensor([[stoi[s] for s in pairs[i]["hist_a"]] for i in ids], dtype=torch.long, device=DEV)
            tb = torch.tensor([[stoi[s] for s in pairs[i]["hist_b"]] for i in ids], dtype=torch.long, device=DEV)
            pa, pb = board_probs(model.decode(ta), kind), board_probs(model.decode(tb), kind)
            for j, i in enumerate(ids):
                mass_a[i] = float(pa[j, pairs[i]["legal_a"]].sum())
                mass_b[i] = float(pb[j, pairs[i]["legal_b"]].sum())
            idx = torch.tensor(ids, device=DEV)
            R[idx] = model.residual_stack(ta)[point][:, -1]
            Rcf[idx] = model.residual_stack(tb)[point][:, -1]
    normal = (mass_a >= LEGAL_MASS) & (mass_b >= LEGAL_MASS)
    keep = np.where(normal)[0]
    counts = {**pair_log, "legal_mass_both": int(len(keep))}
    rows = []
    for i in keep:
        p = pairs[i]
        tiles = np.where(p["changed"])[0]
        rows.append([int(t) * N_CLASSES + int(c) for t in tiles for c in (p["mine_a"][t], p["mine_b"][t])])
    # the covariance corpus: the canonical probe split holds exactly 20000 games, so the last
    # 2000 of them (the brief's fallback; the fit held out 20% of games by sequence)
    n_games = len(data.tokens)
    lo = OTH_PROBE_N if n_games > OTH_PROBE_N + N_COV else n_games - N_COV
    cov_note = (f"cov games {lo}..{lo + N_COV - 1} of {n_games} "
                + ("(beyond the fit corpus)" if lo >= OTH_PROBE_N else "(the last 2000 of the fit corpus)"))
    rows_cov = []
    with torch.no_grad():
        for s in range(lo, lo + N_COV, 250):
            idx = torch.from_numpy(data.tokens[s: s + 250]).to(DEV)
            m = torch.from_numpy(data.mask[s: s + 250]).to(DEV)
            rows_cov.append(model.residual_stack(idx)[point][m])
    cov = torch.cov(zspace(probe, torch.cat(rows_cov)).T)
    del rows_cov
    kidx = torch.from_numpy(keep).to(DEV)
    z, zcf = zspace(probe, R[kidx]), zspace(probe, Rcf[kidx])
    dz = zcf - z
    perm = derangement(len(keep), np.random.default_rng(SEED))
    dgen = z[torch.from_numpy(perm).to(DEV)] - z
    W = probe.net.weight.detach()
    A = haufe_patterns(W, cov)
    res = alignment(dz, dgen, W, A, rows)
    res["_per_case"]["delta_norm_over_h_norm"] = (dz.norm(dim=1) / z.norm(dim=1)).cpu().numpy()
    res["_per_case"]["n_changed_tiles"] = np.array([pairs[i]["n_changed"] for i in keep])
    out = {"run": run_rel, "env": "othello", "instance": inst, "target": "mine/theirs", "point": point,
           "n_cases": int(len(keep)), **{k_: v for k_, v in res.items() if k_ != "_per_case"},
           "counterfactual": (f"one of the last {K_BACK} moves substituted by another legal move, remaining "
                              f"moves replayed (pilot.make_pairs, pool {POOL_OTH}); kept if the model's legal "
                              f"mass >= {LEGAL_MASS} on both histories"),
           "notes": (f"cases {counts}; tiles changed/case mean "
                     f"{np.mean([pairs[i]['n_changed'] for i in keep]):.2f}; {cov_note}")}
    del model
    torch.cuda.empty_cache()
    return out, res["_per_case"]


# ── main ─────────────────────────────────────────────────────────────────────


HDR = (f"{'run':<46} {'target':<14} {'pt':>2} {'n':>4} {'rows':>5} | {'rows':>6} {'gen':>6} {'x':>5} "
       f"| {'haufe':>6} {'genH':>6} {'x':>5}")


def fmt_row(r: dict) -> str:
    return (f"{r['run']:<46} {r['target']:<14} {r['point']:>2} {r['n_cases']:>4} {r['n_rows']:>5.1f} "
            f"| {r['rows_frac']:>6.3f} {r['rows_generic']:>6.3f} {r['rows_ratio']:>4.1f}x "
            f"| {r['haufe_frac']:>6.3f} {r['haufe_generic']:>6.3f} {r['haufe_ratio']:>4.1f}x")


def summary_md(results: list[dict]) -> str:
    lines = ["# Table 3 — alignment of the true edit displacement with the probe's row subspace",
             "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `scripts/table3_alignment.py`. One residual "
             "point per row: the best PI arm's point from the run's `scores.json`. Fractions are "
             "‖Q Qᵀ Δ‖² / ‖Δ‖² in the probe's z-space, mean over cases; `generic` replaces Δ by the "
             "displacement to an unrelated kept case (fixed-point-free permutation, seed 0); `Haufe` "
             "uses the rows of A = Σ Wᵀ (W Σ Wᵀ)⁻¹ instead of W (Σ over 2000 held-out sequences). "
             "`rows` = mean number of probe rows the edit changes per case. No editability here.",
             "",
             "| run | target | point | n cases | rows | rows frac | rows generic | ratio | Haufe frac | Haufe generic | ratio |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['run']} | {r['target']} | {r['point']} | {r['n_cases']} | {r['n_rows']:.1f} "
                     f"| {r['rows_frac']:.3f} | {r['rows_generic']:.3f} | {r['rows_ratio']:.1f}x "
                     f"| {r['haufe_frac']:.3f} | {r['haufe_generic']:.3f} | {r['haufe_ratio']:.1f}x |")
    lines += ["", "## Counterfactual and case counts per row", ""]
    for r in results:
        lines.append(f"- **{r['run']} / {r['target']}** — {r['counterfactual']}. {r['notes']}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings matched against '<run>@<target>' (default: every row)")
    a = ap.parse_args()
    conds = CONDITIONS if not a.only else [c for c in CONDITIONS if any(s in f"{c[0]}@{c[2]}" for s in a.only)]
    (EXP / "scores").mkdir(parents=True, exist_ok=True)
    json_path = EXP / "scores" / "table3_alignment.json"
    prev = {(r["run"], r["target"]): r for r in json.loads(json_path.read_text())} if (a.only and json_path.exists()) else {}
    results, per_case, t_all = [], {}, time.time()
    log(HDR)
    for run_rel, env, target in conds:
        t0 = time.time()
        try:
            row, pc = run_othello(run_rel) if env == "othello" else run_discworld(run_rel, env, target)
        except Exception as e:  # keep going; report the hole
            log(f"{run_rel:<46} {target:<14} FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue
        results.append(row)                       # exactly the contract's keys, nothing else
        per_case[f"{run_rel}@{target}"] = pc
        log(fmt_row(row) + f"   [{(time.time() - t0) / 60:.2f} min]")
    # merge with rows from an earlier full run when --only was used
    done = {(r["run"], r["target"]): r for r in results}
    merged = [done.get((c[0], c[2])) or prev.get((c[0], c[2])) for c in CONDITIONS]
    merged = [r for r in merged if r is not None]
    json_path.write_text(json.dumps(merged, indent=1))
    (EXP / "scores" / "table3_summary.md").write_text(summary_md(merged))
    npz = {f"{k}::{m}": v for k, d in per_case.items() for m, v in d.items()}
    if npz:
        np.savez_compressed(EXP / "scores" / "table3_alignment_cases.npz", **npz)
    log(f"\n{len(results)}/{len(conds)} rows in {(time.time() - t_all) / 60:.1f} min -> "
        f"{json_path.relative_to(REPO)}, {(EXP / 'scores' / 'table3_summary.md').relative_to(REPO)}")


if __name__ == "__main__":
    main()
