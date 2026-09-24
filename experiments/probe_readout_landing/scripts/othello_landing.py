"""Does an Othello edit land in the PROBE's own output space? (2026-09-17)

The discworld side records this (``arms.readout_error`` / ``readout_landed``, the α=1
"the probe reads the target exactly and the frame does not move" check). Othello never
did. This script asks, at each instance's CANONICAL best arm and at PI's α=1 exact-landing
write, whether the probe the editor was built from reads the target board at the edited
residual — the tile the edit asks for, and all 64 squares.

Read-only. Canonical primitives (``pinv_step``, ``probe_direction`` / ``addition_delta``,
``build_edit_spec`` / ``_descend``, ``inverse_overwrite``), canonical cached probes, the
canonical 1000-case bench. PI's best arm is also DECODED and scored so the recomputed
Edit Index can be checked against ``scores.json`` — if that matches, the write measured
here is the canonical write.

    .pim/bin/python experiments/probe_readout_landing/scripts/othello_landing.py --run runs/initial_othello_comparison/L-oth-20m
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.editors.grad_steer import _descend, build_edit_spec  # noqa: E402
from pim.editors.inverse import inverse_overwrite  # noqa: E402
from pim.editors.nanda import addition_delta, probe_direction  # noqa: E402
from pim.editors.pinv import pinv_step, swap_class_logits  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello.bench import case_targets, cases_path, benchmark_from_cases  # noqa: E402
from pim.environments.othello.data import (N_CLASSES, N_TILES, board_probs, canonical_vocab,
                                            flatten_rows, tokens_and_labels)  # noqa: E402
from pim.metrics.set_editability import edit_index_legal, move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402
from pim.probes.inverse import INVERSE_EPOCHS, INVERSE_HIDDEN  # noqa: E402

DEV = "cuda"
EXP = REPO / "experiments/probe_readout_landing"


def readout_board(probe, h: torch.Tensor) -> torch.Tensor:
    """(B, 64) the probe's board labels at h — argmax for a classification probe, the
    sign of the signed mine/theirs value for the regression one."""
    out = probe(h)
    if probe.n_classes is not None:
        return out.view(h.shape[0], N_TILES, N_CLASSES).argmax(-1)
    return out  # regression: caller compares values, not labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    S = json.loads((run_dir / "scores.json").read_text())
    best, st = S["best"], S["settings"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV); model.eval()
    NP = n_points(model)
    stoi = canonical_vocab(); itos = {v: k for k, v in stoi.items()}

    cases = pickle.load(open(cases_path(inst), "rb"))[: a.n]
    bench = benchmark_from_cases([{"history": c["history"], "pos_int": c["pos_int"],
                                   "ori_color": c["ori_color"]} for c in cases], **rules)
    cur_lab, tgt_lab = case_targets(bench)
    n = bench.n_cases

    # the FULL target board per case (mine/theirs frame, the edited tile flipped)
    hist = [None] * n
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels([hist[i] for i in range(n)], **rules)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n)])
    s_post = s_pre.copy(); s_post[np.arange(n), bench.pos_int] = tgt_lab
    assert (s_pre[np.arange(n), bench.pos_int] == cur_lab).all()
    s_post_t = torch.from_numpy(s_post).long().to(DEV)

    ptok, pln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
    ng = int(st["oth_probe_games"])
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:ng], pln[:ng])], **rules)
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}

    # IM's cached g at its best point
    store = ProbeCache(run_dir / "probes")
    seq_of_row, _states = flatten_rows(data, "mine")
    n_seq = int(data.mask.shape[0])
    g_by_point = {}
    if "IM" in best:
        ell = int(best["IM"]["point"])
        fname, prov = store.key(model, kind="inverse_map", target="mine-onehot", n_seq=n_seq,
                                split="sequence", seed=0, hidden=INVERSE_HIDDEN,
                                epochs=INVERSE_EPOCHS, point=int(ell), n_games=ng)
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            g_by_point[ell] = hit["g"].to(DEV)
        else:
            print(f"  (no cached inverse map at point {ell}; IM skipped)", flush=True)

    onehot = np.eye(N_CLASSES, dtype=np.float32)
    Xpost_t = torch.from_numpy(onehot[s_post].reshape(n, -1)).to(DEV)

    # ── the arms to measure: (name, point, alpha, probe family) ────────────────
    jobs = []
    for ed in ("PI", "ND", "GS", "IM"):
        if ed in best:
            b = best[ed]
            jobs.append((ed, int(b["point"]), float(b.get("alpha", 1.0)), "best"))
    if "PI" in best:
        jobs.insert(1, ("PI", int(best["PI"]["point"]), 1.0, "alpha1"))

    res = {"run": a.run, "instance": inst, "n_cases": n, "arms": []}
    for ed, ell, alpha, tag in jobs:
        tile_ok = np.zeros(n, bool); board_ok = np.zeros(n, bool)
        board_hamming = np.zeros(n, float)
        probs = np.zeros((n, N_TILES), np.float32) if ed == "PI" else None
        p_lin, p_mlp = lin[ell], mlp[ell]
        probe_used = p_mlp if ed == "GS" else p_lin
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx = torch.from_numpy(toks).to(DEV)
            bsz = len(ids)
            ar = torch.arange(bsz, device=DEV)
            sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
            td = torch.from_numpy(tgt_lab[ids]).to(DEV)
            cd = torch.from_numpy(cur_lab[ids]).to(DEV)
            with torch.no_grad():
                rs = model.residual_stack(idx)
                h0 = rs[ell][:, -1].clone()
            del rs
            with torch.no_grad():
                if ed == "PI":
                    lg = swap_class_logits(p_lin(h0), sq, cd, td)
                    h1 = h0 + alpha * pinv_step(h0, lg.view(bsz, -1), p_lin, space="zspace")
                elif ed == "ND":
                    d = probe_direction(p_lin, sq * N_CLASSES + td, per_sample=True,
                                        subtract_rows=sq * N_CLASSES + cd)
                    h1 = h0 + addition_delta(h0, d, alpha)
                elif ed == "IM":
                    if ell not in g_by_point:
                        h1 = None
                    else:
                        h1 = inverse_overwrite(g_by_point[ell], Xpost_t[torch.as_tensor(ids, device=DEV)])
                else:  # GS — the canonical descent on the MLP probe at this point
                    h1 = None
            if ed == "GS":
                cm = np.zeros((bsz, N_TILES), bool); cm[np.arange(bsz), bench.pos_int[ids]] = True
                tv = torch.zeros(bsz, N_TILES, dtype=torch.long, device=DEV)
                tv[ar, sq] = td
                spec = build_edit_spec(p_mlp, h0, cm, tv, beta=float(st["oth_gs_beta"]))
                h1 = _descend(p_mlp, h0, spec, alpha, int(st["oth_gs_steps"]))
            if h1 is None:
                tile_ok[ids] = False; board_ok[ids] = False; board_hamming[ids] = np.nan
                continue
            with torch.no_grad():
                lab = readout_board(probe_used, h1)
                tgt_board = s_post_t[torch.as_tensor(ids, device=DEV)]
                if probe_used.n_classes is None:
                    lab = lab  # regression probe: not label-comparable, flagged below
                    tile_ok[ids] = False; board_ok[ids] = False; board_hamming[ids] = np.nan
                else:
                    tile_ok[ids] = (lab[ar, sq] == td).cpu().numpy()
                    same = (lab == tgt_board)
                    board_ok[ids] = same.all(1).cpu().numpy()
                    board_hamming[ids] = (~same).sum(1).float().cpu().numpy()
                if ed == "PI" and probs is not None:
                    def hook(layer, x, _h=h1, _l=ell):
                        if layer != _l:
                            return x
                        out = x.clone(); out[:, -1] = _h
                        return out
                    probs[ids] = board_probs(model.decode(idx, edit=hook),
                                             getattr(model, "output_kind", "logits"))
        rec = {"editor": ed, "tag": tag, "point": ell, "alpha": alpha,
               "probe": "mlp" if ed == "GS" else "linear",
               "tile_landed": float(np.mean(tile_ok)), "board_landed": float(np.mean(board_ok)),
               "board_wrong_squares_mean": float(np.nanmean(board_hamming))}
        if ed == "PI" and probs is not None:
            rec["recomputed_symdiff"] = float(np.nanmean(
                edit_index_legal(probs, bench.legal_pre, bench.legal_post, "symdiff")))
        res["arms"].append(rec)
        extra = f"  recomputed symdiff {rec['recomputed_symdiff']:+.3f}" if "recomputed_symdiff" in rec else ""
        print(f"  {ed:3s} [{tag:6s}] pt{ell} a{alpha:<6g} probe={rec['probe']:6s} "
              f"tile landed {rec['tile_landed']:6.1%}  board landed {rec['board_landed']:6.1%}  "
              f"wrong squares {rec['board_wrong_squares_mean']:.2f}{extra}", flush=True)
    out = EXP / "scores" / f"landing_{Path(a.run).name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
