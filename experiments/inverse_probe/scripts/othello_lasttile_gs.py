"""GS (gradient steering on the MLP probes) on the LAST-TILE case set — the canonical layer
sets and α grid, re-searched — appended as canonical_on_cases["GS"] to the run's
scores/othello_<run>_mirror128_lasttile.json (2026-09-14 evening, Sevan)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import benchmark_from_cases, cases_path  # noqa: E402
from pim.environments.othello.vendor.othello import OthelloBoardState  # noqa: E402
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args(); t0 = time.time()
run = REPO / "runs" / a.run
S = json.loads((run / "scores.json").read_text())["settings"]
inst = json.loads((run / "config.json").read_text())["data"]["instance"]; rules = oc.rules_of(inst)
model, _ = load_checkpoint(run / "best_model.pt", device="cuda"); model.eval(); NP = n_points(model)
import pickle
cases = []
for c in pickle.load(open(cases_path(inst), "rb")):
    h = [int(t) for t in c["history"]]; sq = h[-1]
    b = OthelloBoardState(**rules); b.update(h, prt=False); pre = sorted(b.get_valid_moves())
    ori = 0.0 if b.state[sq // 8, sq % 8] < 0 else 2.0
    pb = OthelloBoardState(**rules); pb.update(h, prt=False); pb.state[sq // 8, sq % 8] = int(2 - ori) - 1
    post = sorted(pb.get_valid_moves())
    if post and post != pre:
        cases.append({"history": h, "pos_int": sq, "ori_color": ori, "game": c.get("game")})
bench = benchmark_from_cases(cases, **rules); cur, tgt = case_targets(bench)
uns = oa.unsteered_probs(model, bench)
tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)["probe"])
itos = {v: k for k, v in canonical_vocab().items()}; n_games = S["oth_probe_games"]
data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n_games], ln[:n_games])], **rules)
grid = oa.fit_probe_grid(model, data, cache_dir=run / "probes", log=None)
mlp = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(NP)}
rows = []
for ls in S["oth_gs_layers"]:
    for al in S["oth_alpha_gs"]:
        pr, card = oa.grad_steer_arm(model, bench, mlp, ls, alpha=al, n_steps=S["oth_gs_steps"], beta=S["oth_gs_beta"], target_labels=tgt)
        rows.append({"point": ls, "alpha": al, "edit_index": card["edit_index_symdiff"], "edit_index_union": card["edit_index_union"],
                     "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post)})
        print(f"GS layers≥{ls} α {al}: {rows[-1]['edit_index']:+.3f}/{rows[-1]['fidelity_ratio']:.2f}  [{(time.time()-t0)/60:.1f} min]", flush=True)
best = max(rows, key=lambda r: r["edit_index_union"]); guarded = [r for r in rows if r["fidelity_ratio"] < 1]
rec = {"best": best, "best_guarded": max(guarded, key=lambda r: r["edit_index"]) if guarded else None, "arms": rows}
p = REPO / "experiments/inverse_probe/scores" / f"othello_{a.run.split('/')[-1]}_mirror128_lasttile.json"
d = json.loads(p.read_text()); assert d["n_cases"] == bench.n_cases
d.setdefault("canonical_on_cases", {})["GS"] = rec; p.write_text(json.dumps(d, indent=1, default=float))
bg = rec["best_guarded"]
gs_txt = "—" if bg is None else f"{bg['edit_index']:+.3f}/{bg['fidelity_ratio']:.2f}"
print(f"GS on last-tile cases: best {best['edit_index']:+.3f}/{best['fidelity_ratio']:.2f} (pt {best['point']}, α {best['alpha']}); guarded {gs_txt} → {p.name}")
