"""Are the CENTRE four tiles editable in standard Othello? (2026-09-11, Sevan; contained.)

The four starting squares (d4 e4 d5 e5 = flat indices 27 28 35 36) are occupied from move 0
and recoloured most often. Li's shipped 1001 never intervene on them, so centre-tile cases
are synthesised from held-out test games with the bench's own recipe (see below); the canonical probes of `L-oth-20m` (cached, nothing refitted)
and the canonical editors / α grids are run on that subset exactly as `master_eval` runs them
on the whole bench. Reports the unedited floor, each editor's best arm (Edit Index / move
fidelity) on the subset, and the same editor at the WHOLE-BENCH best configuration.
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"
RUN = REPO / "runs/initial_othello_comparison/L-oth-20m"
CENTRE = {27, 28, 35, 36}
S = json.loads((RUN / "scores.json").read_text())["settings"]
model, _ = load_checkpoint(RUN / "best_model.pt", device=DEV); model.eval(); NP = n_points(model)
# Li's shipped 1001 never intervene on a centre tile (checked: 0 of 1001), so the cases are
# SYNTHESISED the way `bench.synthesise_cases` makes the other instances' benches — a real
# held-out game prefix, one occupied square flipped, rejected if the legal set is unchanged
# or empty, prefix lengths following the shipped 1001 — with the square drawn from the
# CENTRE four instead of the non-centre occupied squares.
from pim.environments.othello.bench import shipped_length_distribution
from pim.environments.othello.vendor.othello import OthelloBoardState
N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 100
tok_t, ln_t = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("test",))["test"])
itos = {v: k for k, v in canonical_vocab().items()}
hist = [[itos[int(t)] for t in row[:L]] for row, L in zip(tok_t[:5000], ln_t[:5000])]
rng = np.random.default_rng(0); lc = shipped_length_distribution(); tot = sum(lc.values())
quota = {L: int(round(N_CASES * c / tot)) for L, c in sorted(lc.items())}
sub, rej_same, rej_empty = [], 0, 0
for L, want in quota.items():
    got = 0
    for g in rng.permutation([i for i, h in enumerate(hist) if len(h) > L]):
        if got >= want: break
        h = list(hist[g][:L]); board = OthelloBoardState(); board.update(h, prt=False)
        pre = sorted(board.get_valid_moves())
        if not pre: continue
        for sq in rng.permutation(sorted(CENTRE)):
            sq = int(sq); ori = 0.0 if board.state[sq // 8, sq % 8] < 0 else 2.0
            post = OthelloBoardState(); post.update(h, prt=False); post.state[sq // 8, sq % 8] = int(2 - ori) - 1
            lp = sorted(post.get_valid_moves())
            if not lp: rej_empty += 1; continue
            if lp == pre: rej_same += 1; continue
            sub.append({"history": h, "pos_int": sq, "ori_color": ori, "game": int(g)}); got += 1; break
print(f"{len(sub)} synthesised centre-tile cases (tiles {sorted({int(c['pos_int']) for c in sub})}; "
      f"prefix lengths {min(len(c['history']) for c in sub)}–{max(len(c['history']) for c in sub)}; "
      f"rejected same-legal {rej_same}, empty {rej_empty})", flush=True)
bench = benchmark_from_cases(sub)
cur, tgt = case_targets(bench)
u = oa.unsteered(model, bench); uns = oa.unsteered_probs(model, bench)
# the canonical probes, from the run's cache (the data argument only feeds a miss)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
itos = {v: k for k, v in canonical_vocab().items()}
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:S["oth_probe_games"]], ln[:S["oth_probe_games"]])])
grid = oa.fit_probe_grid(model, data, cache_dir=RUN / "probes", log=None)
lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}
arms = []
for mode, label, al in (("add_sub", "ND", S["oth_alpha_nd"]), ("pinv", "PI", S["oth_alpha_pi"])):
    for ell in range(NP):
        for a in al:
            pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=a, points={ell})
            arms.append({"editor": label, "point": ell, "alpha": a,
                         "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                         **{k: v for k, v in card.items() if isinstance(v, (int, float))}})
for ls in S["oth_gs_layers"]:
    for a in S["oth_alpha_gs"]:
        pr, card = oa.grad_steer_arm(model, bench, mlp, ls, alpha=a, n_steps=S["oth_gs_steps"],
                                     beta=S["oth_gs_beta"], target_labels=tgt)
        arms.append({"editor": "GS", "point": ls, "alpha": a,
                     "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                     **{k: v for k, v in card.items() if isinstance(v, (int, float))}})
whole = json.loads((RUN / "scores.json").read_text())["best"]
print(f"\nunedited (centre-tile cases, n={len(sub)}): Edit Index {u['edit_index_union']:+.3f}"
      f"   | whole bench −0.713")
print(f"{'editor':6s} {'best on centre tiles':>34s} | {'same editor at the whole-bench best arm':>44s} | whole-bench best")
out = {"run": str(RUN.relative_to(REPO)), "n_cases": len(sub), "tiles": sorted(CENTRE),
       "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "cases": [{k: (v if k != "history" else len(v)) for k, v in c.items()} for c in sub], "arms": arms, "best": {}}
for ed in ("PI", "ND", "GS"):
    sub_arms = [r for r in arms if r["editor"] == ed]
    b = max(sub_arms, key=lambda r: r["edit_index_union"])
    wb = whole[ed]
    same = [r for r in sub_arms if r["point"] == wb["point"] and abs(r["alpha"] - wb["alpha"]) < 1e-9]
    same = same[0] if same else None
    out["best"][ed] = {"subset_best": b, "at_whole_bench_arm": same, "whole_bench_best": wb}
    print(f"{ed:6s} {b['edit_index_union']:+.3f} / fid {b['fidelity_ratio']:.2f} (pt {b['point']}, α {b['alpha']:g})"
          f"{'':>6s} | {same['edit_index_union'] if same else float('nan'):+.3f} / fid {same['fidelity_ratio'] if same else float('nan'):.2f}"
          f" (pt {wb['point']}, α {wb['alpha']:g}){'':>14s} | {wb['edit_index_union']:+.3f} / {wb['fidelity_ratio']:.2f}")
(REPO / "experiments/othello_centre_tiles/scores/centre_tiles_L-oth-20m.json").write_text(json.dumps(out, indent=1, default=float))
