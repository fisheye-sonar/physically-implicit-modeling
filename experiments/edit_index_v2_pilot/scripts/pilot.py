#!/usr/bin/env python
"""PILOT (2026-09-11): the model-referenced Edit Index — both references are the MODEL'S OWN
predictions: p_A on the real history, p_B on a paired COUNTERFACTUAL history whose simulator
state differs from A by one edit. Every edited output is scored against (p_A, p_B) on the
simulator's differing support:

    EI_v2  = (d(pred, p_A) − d(pred, p_B)) / (d(pred, p_A) + d(pred, p_B))     [-1 unedited, +1 = p_B]
    guard2 = RMSE(pred, p_B) / RMSE(p_A, p_B)   over the whole frame / all 64 squares

beside the CANONICAL construction on the same cases (EI_v1 against the simulator's two
references; the guard against S(B)), and the CEILING (p_B under the canonical index).

Paired histories. Discworld: the edited object's whole trajectory shifted by the teleport
vector, other object untouched, same noise draws (= ``arms.counterfactual_history``); kept only
if in-frustum and collision-free at every frame. Othello: ONE of the last k moves substituted
by another legal move and the original remaining moves replayed; kept only if the replay is
legal, the mover is unchanged, the board differs, and the legal set differs.

Filters (Sevan, 2026-09-11): a case counts only if the model's predictions on the two
histories differ by a sufficient amount — RMSE(p_A, p_B) on the support ≥ max(ABS_FLOOR,
REL_FLOOR × the simulator's own separation on that support) — and, on Othello, the model
treats both histories as ordinary games (legal mass ≥ LEGAL_MASS on each).

One arm per editor, at the run's CANONICAL best (point, α, dims) from its scores.json — no
sweep. A few dozen cases per condition. Writes scores/<condition>.json (+ per-case arrays),
scores/summary.json, outputs/pilot_ei.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook  # noqa: E402
from pim.editors.nanda import addition_delta, addition_hook, probe_direction  # noqa: E402
from pim.editors.oracle_overwrite import overwrite_rollout  # noqa: E402
from pim.editors.pinv import pinv_step, swap_class_logits  # noqa: E402
from pim.environments import layout  # noqa: E402
from pim.metrics.edit_index import edit_index_per_case, masked_rmse_per_case  # noqa: E402
from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio, uniform_over_legal  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

EXP = _REPO / "experiments" / "edit_index_v2_pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
ABS_FLOOR = {"frames": 0.02, "dist": 0.01}      # RMSE(p_A, p_B) on the support, absolute floor
REL_FLOOR = 0.25                                 # … and ≥ this fraction of the simulator's separation
LEGAL_MASS = 0.98                                # Othello: the model treats both histories as normal
K_BACK = 4                                       # Othello: substitute one of the last k moves
N_TARGET = 48                                    # valid cases aimed for per condition
POOL_OTH = 900                                   # Othello pair attempts drawn from the bench histories
GS_STEPS, GS_BETA = 100, 0.2                     # the scorer's GS settings
_CURRENT = "unnamed"                             # the condition being scored (for the arrays dump)

CONDITIONS = {
    # name: (run dir, env, probe target block, notes)
    "L-oth-20m": ("initial_othello_comparison/L-oth-20m", "othello", "mine/theirs"),
    "L-oth-adjacent-flip-20m": ("adjacent_flip_ablation/L-oth-adjacent-flip-20m", "othello", "mine/theirs"),
    "L-oth-adjacent-20m": ("adjacency_ablation/L-oth-adjacent-20m", "othello", "mine/theirs"),
    "L-oth-noflip-20m": ("flip_ablation/L-oth-noflip-20m", "othello", "mine/theirs"),
    "L-dw-noiseless-20m": ("noise_ablation/L-dw-noiseless-20m", "discworld", "frustum"),
    "L-dw-8ray-20m": ("ray_ablation/L-dw-8ray-20m", "discworld", "frustum"),
    "L-dw-8ray-tok-20m": ("interface_ablation/L-dw-8ray-tok-20m", "discworld-tokens", "frustum"),
    "L-dw-8ray-20m@appearance-fac": ("ray_ablation/L-dw-8ray-20m", "discworld", "appearance-fac"),
    "L-dw-8ray-tok-20m@appearance-fac": ("interface_ablation/L-dw-8ray-tok-20m", "discworld-tokens", "appearance-fac"),
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _rmse_all(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).mean(1))


def _summ(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75)),
            "min": float(x.min()), "max": float(x.max())}


def best_arms(run_dir: Path, env: str, block: str) -> dict:
    """{editor: (point, alpha, dims)} — the canonical best arm per editor from scores.json."""
    s = json.loads((run_dir / "scores.json").read_text())
    best = s["best"] if env == "othello" else s["bases"][block]["best"]
    out = {}
    for ed in ("PI", "ND", "GS"):
        b = best.get(ed)
        if b:
            out[ed] = (int(b["point"]), float(b["alpha"]), b.get("dims", "all"))
    return out


def score_block(pred: dict, p_a: np.ndarray, p_b: np.ndarray, ref_a: np.ndarray, ref_b: np.ndarray,
                supp: np.ndarray, keep: np.ndarray, kind: str, guard_v1) -> dict:
    """Both constructions on the kept cases. ``pred`` {editor: (N, D)}; refs (N, D); supp (N, D) bool."""
    out = {"n": int(keep.sum())}
    sep_model = masked_rmse_per_case(p_a, p_b, supp)
    sep_sim = masked_rmse_per_case(ref_a, ref_b, supp)
    out["separation_model"] = _summ(sep_model[keep])
    out["separation_sim"] = _summ(sep_sim[keep])
    out["separation_ratio"] = _summ((sep_model / np.maximum(sep_sim, 1e-12))[keep])
    # the ceiling: p_B under the canonical construction (and, trivially, +1 under v2)
    out["ceiling_v1"] = float(np.nanmean(edit_index_per_case(p_b, ref_b, ref_a, supp)[keep]))
    out["unedited_v1"] = float(np.nanmean(edit_index_per_case(p_a, ref_b, ref_a, supp)[keep]))
    out["unedited_v2"] = float(np.nanmean(edit_index_per_case(p_a, p_b, p_a, supp)[keep]))
    out["ceiling_v2"] = float(np.nanmean(edit_index_per_case(p_b, p_b, p_a, supp)[keep]))   # +1 by construction
    # raw arrays for audits (the per-case decomposition of the two constructions)
    (EXP / "scores" / "arrays").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXP / "scores" / "arrays" / f"{_CURRENT.replace('@', '_')}.npz", p_a=p_a, p_b=p_b,
                        ref_a=ref_a, ref_b=ref_b, supp=supp, keep=keep, **{f"pred_{ed}": pr for ed, pr in pred.items()})
    out["editors"] = {}
    for ed, pr in pred.items():
        ei2 = edit_index_per_case(pr, p_b, p_a, supp)
        ei1 = edit_index_per_case(pr, ref_b, ref_a, supp)
        g2 = float(_rmse_all(pr[keep], p_b[keep]).mean() / max(_rmse_all(p_a[keep], p_b[keep]).mean(), 1e-12))
        out["editors"][ed] = {
            "ei_v2": float(np.nanmean(ei2[keep])), "ei_v2_per_case": _summ(ei2[keep]),
            "ei_v1": float(np.nanmean(ei1[keep])), "ei_v1_per_case": _summ(ei1[keep]),
            "guard_v2": g2, "guard_v1": float(guard_v1(pr, keep)),
            "frac_ceiling": float(np.nanmean(ei1[keep]) / out["ceiling_v1"]) if out["ceiling_v1"] > 0 else None,
            "_per_case": {"ei_v2": ei2[keep].tolist(), "ei_v1": ei1[keep].tolist()},
        }
    return out


# ── discworld (frames and tokens) ────────────────────────────────────────────


def run_discworld(name: str, run_rel: str, env: str, block: str) -> dict:
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld import bench as dwb
    from pim.environments.discworld.sim import fully_in_frustum
    from pim.metrics.zone_editability import sim_config_from

    run_dir = _REPO / "runs" / run_rel
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg["data"]["instance"]
    tokens = env == "discworld-tokens"
    target = "full" if block == "frustum" else block
    model, info = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    arms = best_arms(run_dir, "discworld", block)
    recipe = dwa.probe_recipe(target, inst, n_seq=30_000)
    enc = {}
    if tokens:
        from pim.environments.discworld import token_bench as tkb
        from pim.environments.discworld.tokens import UNK, FrameVocab, encode
        vocab = FrameVocab.load(run_dir / "vocab.npz")
        _e, _tag = tkb.token_encoder(vocab)
        enc = {"encoder": _e, "encoder_tag": _tag}
    lin = dwa.fit_probes(model, target=target, family="linear", basis_name="frustum",
                         cache_dir=run_dir / "probes", log=None, require_cached=True, **recipe, **enc)
    mlp = dwa.fit_probes(model, target=target, family="mlp", basis_name="frustum",
                         cache_dir=run_dir / "probes", log=None, require_cached=True, **recipe, **enc)
    log(f"  probes: cache hits ({len(lin)} points)  arms {arms}")

    # the canonical bench (192 cases, the block's own case rule). The frame model gets the
    # warmed Bench; the token model cannot warm a frame state (categorical head), so it gets
    # the model-free arrays wrapped with the fields counterfactual_history reads.
    if not tokens:
        b = dwb.load_bench(model, n=192, target=target, basis_name="frustum", instance=inst)
    else:
        from types import SimpleNamespace
        arr = dwb.bench_arrays(n=192, target=target, basis_name="frustum", instance=inst)
        b = SimpleNamespace(obs=arr["obs"], pos=arr["pos"], vel=arr["vel"], edit_object=arr["edit_object"],
                            sim=arr["sim"], zones=arr["zones"], n=arr["n"])
    n = b.n
    ar = np.arange(n)
    k = b.edit_object.astype(int)
    sim = b.sim
    cfg_sim = sim_config_from(sim, dwb.N_OBJ)
    dt = float(sim["dt"])
    delta = b.pos[ar, dwb.EF, k] - (b.pos[ar, dwb.EF - 1, k] + b.vel[ar, dwb.EF - 1, k] * dt)
    cf_pos = b.pos[:, : dwb.EF].copy()
    cf_pos[ar, :, k] += delta[:, None, :]
    min_sep = cfg_sim.collision_margin * 2.0 * cfg_sim.radius
    valid = np.array([fully_in_frustum(cf_pos[i], cfg_sim.radius, cfg_sim)
                      and (np.linalg.norm(cf_pos[i, :, 0] - cf_pos[i, :, 1], axis=-1) >= min_sep).all()
                      for i in range(n)])
    cf = dwa.counterfactual_history(b, noise_matched=True)               # (n, EF, R)
    filters = {"cases": int(n), "valid_counterfactual": int(valid.sum())}

    supp = b.zones.differing                                             # rays where S(A) ≠ S(B)
    ref_a, ref_b = b.zones.gt_unedited, b.zones.gt_edited
    if not tokens:
        with torch.no_grad():
            p_a = dwa.unsteered_rollout(model, b)[:, 0]
            p_b = overwrite_rollout(model, torch.from_numpy(cf).float().to(DEV), dwb.K_ROLL).cpu().numpy()[:, 0]
        pred = {}
        for ed, (pt, a, dims) in arms.items():
            if ed == "PI":
                roll = dwa.pinv_rollout(model, b, lin[pt][0], pt, a, space="zspace", dims=dims)
            elif ed == "ND":
                roll = dwa.nanda_rollout(model, b, lin[pt][0], pt, a, dims=dims)
            else:
                roll = dwa.grad_steer_rollout(model, b, mlp, pt, a, n_steps=GS_STEPS, beta=GS_BETA, dims=dims)
            pred[ed] = np.asarray(roll)[:, 0]
        keep = valid & (supp.sum(1) >= 2)
        filters["support_ge_2_rays"] = int((valid & (supp.sum(1) >= 2)).sum())
        sep_model = masked_rmse_per_case(p_a, p_b, supp)
        sep_sim = masked_rmse_per_case(ref_a, ref_b, supp)
        keep &= sep_model >= np.maximum(ABS_FLOOR["frames"], REL_FLOOR * sep_sim)
        filters["model_separation"] = int(keep.sum())
        guard_v1 = lambda pr, kp: _rmse_all(pr[kp], ref_b[kp]).mean() / _rmse_all(p_a[kp], ref_b[kp]).mean()  # noqa: E731
        kind = "frames"
    else:
        tb = tkb.load_token_bench(vocab, n=192, target=target, basis_name="frustum", instance=inst)
        assert tb.n == n
        tok_b = encode(cf, vocab).astype(np.int64)                        # (n, EF)
        valid &= (tok_b != UNK).all(1) & tb.keep
        filters["valid_counterfactual_in_vocab"] = int(valid.sum())
        with torch.no_grad():
            p_a = tkb.probs_at_edit(model, tb)
            p_b = tkb.frame_probs(model.decode(torch.from_numpy(tok_b).to(DEV)),
                                  getattr(model, "output_kind", "logits")).cpu().numpy()
        V = p_a.shape[1]
        supp = np.zeros((n, V), bool)
        supp[ar, tb.pre_tok] = True
        supp[ar, tb.post_tok] = True
        ref_a = np.zeros((n, V), np.float32); ref_a[ar, tb.pre_tok] = 1.0     # one-hot: the frame-set refs
        ref_b = np.zeros((n, V), np.float32); ref_b[ar, tb.post_tok] = 1.0
        x0 = tkb.residuals_last(model, tb)
        pred = {}
        for ed, (pt, a, dims) in arms.items():
            if ed == "PI":
                h0 = x0[pt]
                tgt = dwa.pinv_target(lin[pt][0], h0, tb)
                h = h0 + a * pinv_step(h0, tgt, lin[pt][0], space="zspace", dims=dwb.dim_idx(dims))
                pr = tkb.probs_at_edit(model, tb, hook=tkb._write_hook(pt, h))
            elif ed == "ND":
                probe = lin[pt][0]
                if tb.kind == "classification":
                    d = dwa.categorical_direction(probe, tb)
                else:
                    idx = dwb.dim_idx(dims)
                    rows = tb.out_dims if idx is None else [r for r in tb.out_dims if r in set(idx)]
                    d = probe_direction(probe, rows)
                pr = tkb.probs_at_edit(model, tb, hook=addition_hook(pt, d, a))
            else:
                pts = {e: mlp[e][0] for e in mlp if e >= pt}
                cm = dwb.restrict_mask(tb.change_mask, dims)
                specs = {e: build_edit_spec(p_, x0[e], cm, tb.tgt, beta=GS_BETA) for e, p_ in pts.items()}
                hook = make_intervention_hook(pts, specs, pt, alpha=a, n_steps=GS_STEPS)
                pr = tkb.probs_at_edit(model, tb, hook=hook)
            pred[ed] = np.asarray(pr)
        keep = valid.copy()
        sep_model = masked_rmse_per_case(p_a, p_b, supp)
        sep_sim = masked_rmse_per_case(ref_a, ref_b, supp)
        keep &= sep_model >= np.maximum(ABS_FLOOR["dist"], REL_FLOOR * sep_sim)
        filters["model_separation"] = int(keep.sum())
        guard_v1 = lambda pr, kp: move_fidelity_ratio(pr[kp], p_a[kp], [[int(t)] for t in tb.post_tok[kp]])  # noqa: E731
        kind = "dist"
    keep_idx = np.where(keep)[0][: (N_TARGET if N_TARGET > 0 else None)]
    keep_final = np.zeros(n, bool); keep_final[keep_idx] = True
    filters["scored"] = int(keep_final.sum())
    res = score_block(pred, p_a, p_b, ref_a, ref_b, supp, keep_final, kind, guard_v1)
    res.update({"condition": name, "run": run_rel, "instance": inst, "env": env, "block": block,
                "arms": {ed: {"point": pt, "alpha": a, "dims": d} for ed, (pt, a, d) in arms.items()},
                "filters": filters, "counterfactual": "trajectory of the edited object shifted by the "
                "teleport vector over frames 0..EF-1 (arms.counterfactual_history), noise-matched",
                "support": "rays where the two clean renders differ" if kind == "frames"
                else "the two worlds' frame tokens (union)"})
    return res


# ── othello ──────────────────────────────────────────────────────────────────


def mine_board(board) -> np.ndarray:
    from pim.environments.othello.data import BLANK, MINE, THEIRS
    st = (board.state + 1).flatten().astype(np.int64)
    nxt = 2 if board.next_hand_color > 0 else 0
    return np.where(st == 1, BLANK, np.where(st == nxt, MINE, THEIRS)).astype(np.int64)


def make_pairs(histories: list[list[int]], rules: dict, k: int, pool: int, rng) -> list[dict]:
    from pim.environments.othello.data import BLANK
    from pim.environments.othello.vendor.othello import OthelloBoardState

    pairs, tried, rejected = [], 0, {"no_alternative": 0, "illegal_replay": 0, "mover_changed": 0,
                                     "board_same": 0, "legal_same": 0}
    order = rng.permutation(len(histories))
    for i in order:
        if tried >= pool:
            break
        h = list(histories[i])
        L = len(h)
        if L < 2:
            continue
        tried += 1
        j = int(rng.integers(1, min(k, L - 1) + 1))          # substitute move L-j (1 = the last move)
        pre = OthelloBoardState(**rules)
        pre.update(h[: L - j], prt=False)
        alts = [m for m in pre.get_valid_moves() if m != h[L - j]]
        if not alts:
            rejected["no_alternative"] += 1
            continue
        alt = int(rng.choice(alts))
        hb = h[: L - j] + [alt] + h[L - j + 1:]
        bb = OthelloBoardState(**rules)
        try:
            bb.update(hb, prt=False)
        except AssertionError:
            rejected["illegal_replay"] += 1
            continue
        ba = OthelloBoardState(**rules)
        ba.update(h, prt=False)
        if bb.next_hand_color != ba.next_hand_color:
            rejected["mover_changed"] += 1
            continue
        ma, mb = mine_board(ba), mine_board(bb)
        changed = ma != mb
        if not changed.any():
            rejected["board_same"] += 1
            continue
        la, lb = sorted(ba.get_valid_moves()), sorted(bb.get_valid_moves())
        if la == lb:
            rejected["legal_same"] += 1
            continue
        occ = int(((ma == BLANK) != (mb == BLANK)).sum())        # occupancy changes
        pairs.append({"hist_a": h, "hist_b": hb, "j": j, "mine_a": ma, "mine_b": mb, "changed": changed,
                      "n_changed": int(changed.sum()), "n_occupancy": occ, "n_colour": int(changed.sum() - occ),
                      "legal_a": la, "legal_b": lb})
    return pairs, {"attempts": tried, "pairs": len(pairs), "rejected": rejected}


def tile_stats(pairs: list[dict]) -> dict:
    nc = np.array([p["n_changed"] for p in pairs])
    out = {"n_pairs": len(pairs), "tiles_changed": _summ(nc),
           "occupancy_changes": _summ([p["n_occupancy"] for p in pairs]),
           "colour_changes": _summ([p["n_colour"] for p in pairs]),
           "histogram": {int(v): int(c) for v, c in zip(*np.unique(nc, return_counts=True))},
           "by_j": {int(j): _summ(nc[[p["j"] == j for p in pairs]]) for j in sorted({p["j"] for p in pairs})}}
    return out


def run_othello(name: str, run_rel: str, block: str) -> dict:
    from pim.environments.othello import arms as oa
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.bench import load_benchmark
    from pim.environments.othello.data import N_CLASSES, N_TILES, board_probs, canonical_vocab

    run_dir = _REPO / "runs" / run_rel
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg["data"]["instance"]
    rules = oc.rules_of(inst)
    model, info = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    arms = best_arms(run_dir, "othello", block)
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}
    kind = getattr(model, "output_kind", "logits")

    # probes: the run's cached mine/theirs grid (same 20k probe games → cache hit)
    paths = oc.build(only=("probe",), instance=inst, log=lambda s: None)
    data = oc.probe_data(paths["probe"], 20_000, **rules)
    t0 = time.time()
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    npts = model.n_layers + 1
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npts)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(npts)}
    log(f"  probes: grid in {time.time() - t0:.0f}s (a cache hit is seconds)  arms {arms}")

    # paired histories from the instance's bench histories
    bench = load_benchmark(inst)
    hists = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row in toks:
            hists.append([itos[int(t)] for t in row])
    rng = np.random.default_rng(SEED)
    pairs, pair_log = make_pairs(hists, rules, K_BACK, POOL_OTH, rng)
    stats_all = tile_stats(pairs)
    log(f"  pairs: {pair_log}  tiles changed {stats_all['tiles_changed']}")

    # the model on both histories, grouped by length (the hooks write x[:, -1])
    n = len(pairs)
    p_a = np.zeros((n, N_TILES), np.float32)
    p_b = np.zeros((n, N_TILES), np.float32)
    pred = {ed: np.zeros((n, N_TILES), np.float32) for ed in arms}
    by_len: dict[int, list[int]] = {}
    for i, p in enumerate(pairs):
        by_len.setdefault(len(p["hist_a"]), []).append(i)
    max_changed = max(p["n_changed"] for p in pairs)
    for L, members in sorted(by_len.items()):
        ids = np.array(members)
        ta = torch.tensor([[stoi[s] for s in pairs[i]["hist_a"]] for i in ids], dtype=torch.long, device=DEV)
        tb_ = torch.tensor([[stoi[s] for s in pairs[i]["hist_b"]] for i in ids], dtype=torch.long, device=DEV)
        with torch.no_grad():
            p_a[ids] = board_probs(model.decode(ta), kind)
            p_b[ids] = board_probs(model.decode(tb_), kind)
        bsz = len(ids)
        cur_b = torch.tensor(np.stack([pairs[i]["mine_a"] for i in ids]), device=DEV)      # (B, 64)
        tgt_b = torch.tensor(np.stack([pairs[i]["mine_b"] for i in ids]), device=DEV)
        chg = torch.tensor(np.stack([pairs[i]["changed"] for i in ids]), device=DEV)      # (B, 64) bool
        # per-case changed tiles, padded with no-op swaps (tile 0, cur == tgt)
        tiles = torch.zeros(bsz, max_changed, dtype=torch.long, device=DEV)
        c_cls = torch.zeros(bsz, max_changed, dtype=torch.long, device=DEV)
        t_cls = torch.zeros(bsz, max_changed, dtype=torch.long, device=DEV)
        for r in range(bsz):
            w = torch.where(chg[r])[0]
            tiles[r, : len(w)] = w
            c_cls[r, : len(w)] = cur_b[r, w]
            t_cls[r, : len(w)] = tgt_b[r, w]
            if len(w) < max_changed:                                  # padding: a no-op swap at tile 0
                c_cls[r, len(w):] = cur_b[r, 0]
                t_cls[r, len(w):] = cur_b[r, 0]
        for ed, (pt, a, _dims) in arms.items():
            if ed in ("PI", "ND"):
                probe = lin[pt]

                def hook(layer, x, _pt=pt, _a=a, _ed=ed, _probe=probe):
                    if layer != _pt:
                        return x
                    cur = x[:, -1]
                    if _ed == "ND":       # target−current contrast at EVERY changed tile, summed
                        W = _probe.net.weight.detach() / _probe.x_std
                        d = (W[tiles * N_CLASSES + t_cls] - W[tiles * N_CLASSES + c_cls]).sum(1)
                        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        delta = addition_delta(cur, d, _a)
                    else:                 # PI: the probe's own read-out with cur ↔ tgt swapped at every changed tile
                        lg = _probe(cur)
                        for m in range(max_changed):
                            lg = swap_class_logits(lg, tiles[:, m], c_cls[:, m], t_cls[:, m])
                        delta = _a * pinv_step(cur, lg.reshape(bsz, -1), _probe, space="zspace")
                    out = x.clone()
                    out[:, -1] = cur + delta
                    return out
                with torch.no_grad():
                    pred[ed][ids] = board_probs(model.decode(ta, edit=hook), kind)
            else:
                with torch.no_grad():
                    rs = model.residual_stack(ta)
                x0 = {e: rs[e][:, -1] for e in range(npts)}
                pts = {e: mlp[e] for e in range(pt, npts)}
                specs = {e: build_edit_spec(mlp[e], x0[e], chg, tgt_b, beta=GS_BETA) for e in pts}
                hook = make_intervention_hook(pts, specs, pt, alpha=a, n_steps=GS_STEPS)
                with torch.no_grad():
                    pred[ed][ids] = board_probs(model.decode(ta, edit=hook), kind)
                del rs, x0, specs
    # references and support (rules): uniform-over-legal A / B on the union of the legal sets
    la = [p["legal_a"] for p in pairs]
    lb = [p["legal_b"] for p in pairs]
    ref_a = np.stack([uniform_over_legal(L_, N_TILES) for L_ in la])
    ref_b = np.stack([uniform_over_legal(L_, N_TILES) for L_ in lb])
    supp = np.zeros((n, N_TILES), bool)
    for i in range(n):
        supp[i, sorted(set(la[i]) | set(lb[i]))] = True
    mass_a = np.array([p_a[i, la[i]].sum() for i in range(n)])
    mass_b = np.array([p_b[i, lb[i]].sum() for i in range(n)])
    normal = (mass_a >= LEGAL_MASS) & (mass_b >= LEGAL_MASS)
    sep_model = masked_rmse_per_case(p_a, p_b, supp)
    sep_sim = masked_rmse_per_case(ref_a, ref_b, supp)
    enough = sep_model >= np.maximum(ABS_FLOOR["dist"], REL_FLOOR * sep_sim)
    keep = normal & enough
    filters = {**pair_log, "model_normal_both": int(normal.sum()), "model_separation": int((normal & enough).sum()),
               "legal_mass_a": _summ(mass_a), "legal_mass_b": _summ(mass_b)}
    keep_idx = np.where(keep)[0][: (N_TARGET if N_TARGET > 0 else None)]     # N_TARGET <= 0: every valid case
    keep_final = np.zeros(n, bool); keep_final[keep_idx] = True
    filters["scored"] = int(keep_final.sum())
    guard_v1 = lambda pr, kp: move_fidelity_ratio(pr[kp], p_a[kp], [lb[i] for i in np.where(kp)[0]])  # noqa: E731
    res = score_block(pred, p_a, p_b, ref_a, ref_b, supp, keep_final, "dist", guard_v1)
    # per scored case: the edit magnitude (tiles changed), for the by-magnitude view
    res["per_case"] = {"n_changed": [pairs[i]["n_changed"] for i in keep_idx],
                       "n_colour": [pairs[i]["n_colour"] for i in keep_idx],
                       "j": [pairs[i]["j"] for i in keep_idx],
                       "ceiling_v1": edit_index_per_case(p_b, ref_b, ref_a, supp)[keep_final].tolist(),
                       "guard_v2_per_case": {ed: (_rmse_all(pr[keep_final], p_b[keep_final]) /
                                                  np.maximum(_rmse_all(p_a[keep_final], p_b[keep_final]), 1e-12)).tolist()
                                             for ed, pr in pred.items()}}
    # the canonical legal-set index on the same cases, for the record (same numbers as ei_v1)
    for ed, pr in pred.items():
        res["editors"][ed]["ei_legal_union"] = float(np.nanmean(edit_index_legal(pr, la, lb, "union")[keep_final]))
    res.update({"condition": name, "run": run_rel, "instance": inst, "env": "othello", "block": block,
                "arms": {ed: {"point": pt, "alpha": a} for ed, (pt, a, _) in arms.items()},
                "filters": filters, "tile_stats_all_pairs": stats_all,
                "tile_stats_scored": tile_stats([pairs[i] for i in keep_idx]),
                "counterfactual": f"one of the last {K_BACK} moves substituted by another legal move, the "
                "original remaining moves replayed; kept if legal, same mover, board and legal set differ",
                "support": "union of the two legal sets"})
    return res


# ── main ─────────────────────────────────────────────────────────────────────


def print_table(results: list[dict]) -> None:
    print(f"\n{'condition':<34}{'n':>4} {'ceil_v1':>8} {'uned_v1':>8} | " +
          " ".join(f"{ed+'_v2':>7} {ed+'_v1':>7} {'g2':>5} {'g1':>5} |" for ed in ("PI", "ND", "GS")))
    for r in results:
        row = f"{r['condition']:<34}{r['n']:>4} {r['ceiling_v1']:>+8.3f} {r['unedited_v1']:>+8.3f} | "
        for ed in ("PI", "ND", "GS"):
            e = r["editors"].get(ed)
            row += (f"{e['ei_v2']:>+7.3f} {e['ei_v1']:>+7.3f} {e['guard_v2']:>5.2f} {e['guard_v1']:>5.2f} |"
                    if e else f"{'—':>7} {'—':>7} {'—':>5} {'—':>5} |")
        print(row)
    print("\nv2: both references = the model's own predictions (unedited → −1 by construction, p_B → +1);"
          " v1: the canonical simulator references on the SAME cases; g = guard (edited vs its +1 reference,"
          " ratio to unedited; >1 degraded); ceil_v1 = p_B under the canonical index.")


def figure(results: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eds = ("PI", "ND", "GS")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(results) + 2.5), sharey=True)
    ys = np.arange(len(results))[::-1]
    for ax, key, title in zip(axes, ("ei_v2", "ei_v1"), ("model-referenced Edit Index (v2)", "canonical Edit Index (v1), same cases")):
        for yi, r in zip(ys, results):
            for e_i, ed in enumerate(eds):
                e = r["editors"].get(ed)
                if not e:
                    continue
                pts = np.asarray(e["_per_case"][key])
                jitter = (np.random.default_rng(0).random(len(pts)) - 0.5) * 0.18
                ax.scatter(pts, yi + (e_i - 1) * 0.25 + jitter, s=9, alpha=0.35, color=f"C{e_i}", label=ed if yi == ys[0] else None)
                ax.plot([np.nanmean(pts)] * 2, [yi + (e_i - 1) * 0.25 - 0.1, yi + (e_i - 1) * 0.25 + 0.1], color=f"C{e_i}", lw=2.2)
            if key == "ei_v1":
                ax.plot([r["ceiling_v1"]] * 2, [yi - 0.42, yi + 0.42], color="k", lw=1.2, ls="--", label="ceiling (p_B)" if yi == ys[0] else None)
                ax.plot([r["unedited_v1"]] * 2, [yi - 0.42, yi + 0.42], color="grey", lw=1.2, label="unedited floor" if yi == ys[0] else None)
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Edit Index per case (points) and mean (bar)")
        ax.grid(True, axis="x", color="#eee")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([f"{r['condition']}  (n={r['n']})" for r in results], fontsize=8)
    fig.suptitle("Pilot — model-referenced Edit Index on paired counterfactual histories, best canonical arm per editor "
                 f"({time.strftime('%Y-%m-%d')})", fontsize=11)
    out = EXP / "outputs" / "pilot_ei.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


def aggregate() -> None:
    """Table + figure + summary.json from every scores/<condition>.json on disk."""
    results = []
    for name in CONDITIONS:
        p = EXP / "scores" / f"{name.replace('@', '_')}.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
    print_table(results)
    summary = [{k: v for k, v in r.items() if k != "editors"} |
               {"editors": {ed: {kk: vv for kk, vv in e.items() if kk != "_per_case"} for ed, e in r["editors"].items()}}
               for r in results]
    (EXP / "scores" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"figure -> {figure(results).relative_to(_REPO)}   ({len(results)} conditions)")


def main() -> None:
    global N_TARGET
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="condition names (default: all)")
    ap.add_argument("--n-target", type=int, default=N_TARGET)
    ap.add_argument("--aggregate", action="store_true", help="only rebuild the table/figure from scores/")
    a = ap.parse_args()
    N_TARGET = a.n_target
    (EXP / "scores").mkdir(parents=True, exist_ok=True)
    if a.aggregate:
        aggregate()
        return
    names = a.only or list(CONDITIONS)
    results = []
    for name in names:
        run_rel, env, block = CONDITIONS[name]
        t0 = time.time()
        global _CURRENT
        _CURRENT = name
        log(f"\n=== {name}  ({run_rel}, block {block}) ===")
        res = run_othello(name, run_rel, block) if env == "othello" else run_discworld(name, run_rel, env, block)
        res["minutes"] = round((time.time() - t0) / 60, 2)
        (EXP / "scores" / f"{name.replace('@', '_')}.json").write_text(json.dumps(res, indent=1, default=float))
        results.append(res)
        e = {ed: (round(v["ei_v2"], 3), round(v["ei_v1"], 3), round(v["guard_v2"], 2)) for ed, v in res["editors"].items()}
        log(f"  n={res['n']}  filters {res['filters']}\n  ceiling_v1 {res['ceiling_v1']:+.3f}  unedited_v1 {res['unedited_v1']:+.3f}"
            f"  editors (ei_v2, ei_v1, guard_v2): {e}  [{res['minutes']} min]")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print_table(results)
    if len(results) == len(CONDITIONS) or a.only is None:
        summary = [{k: v for k, v in r.items() if k != "editors"} |
                   {"editors": {ed: {kk: vv for kk, vv in e.items() if kk != "_per_case"} for ed, e in r["editors"].items()}}
                   for r in results]
        (EXP / "scores" / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"figure -> {figure(results).relative_to(_REPO)}")


if __name__ == "__main__":
    main()
