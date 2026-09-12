"""⚠ SUPERSEDED by honesty_check_v2.py the same day: the legal-mass filter used here is toothless on
adjacency instances (mass 1.000 on everything) and let swap-built histories through. Kept for the record.

Head-to-head on the SAME clean cases (2026-09-11): the true counterfactual history's Edit Index
(the ceiling) vs the run's canonical best PI and ND arms (point/alpha from its scores.json), per
`research/GOTCHAS.md` 2026-09-09 ("report the ceiling beside any patch result"). Reuses
`edit_direction_alignment/scripts/othello_alignment.py::search_cf` and `pim.environments.othello.arms.linear_arm`.
Self-contained; output → scores/honesty_check_<run>.json.
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
sys.path.insert(0, str(REPO / "experiments/edit_direction_alignment/scripts"))
from othello_alignment import replay, search_cf  # noqa: E402
from pim.environments.othello import arms as oa, corpus as oc  # noqa: E402
from pim.environments.othello import case_targets  # noqa: E402
from pim.environments.othello.bench import (
    benchmark_from_cases,
    cases_path,
)  # noqa: E402
from pim.environments.othello.data import (
    board_probs,
    canonical_vocab,
    tokens_and_labels,
)  # noqa: E402
from pim.metrics.set_editability import move_scorecard  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402

DEV = "cuda"
EXP = REPO / "experiments/adjacent_flip_ablation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=900)
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    rules = oc.rules_of(inst)
    S = json.loads((run_dir / "scores.json").read_text())
    best = S["best"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    stoi = canonical_vocab()
    itos = {v: k for k, v in stoi.items()}

    def pred(hists):
        out = np.zeros((len(hists), 64), np.float32)
        by = {}
        for i, h in enumerate(hists):
            by.setdefault(len(h), []).append(i)
        with torch.no_grad():
            for L, ids in by.items():
                idx = torch.from_numpy(
                    np.array([[stoi[x] for x in hists[i]] for i in ids])
                ).to(DEV)
                out[ids] = board_probs(
                    model.decode(idx), getattr(model, "output_kind", "logits")
                )
        return out

    cases = pickle.load(open(cases_path(inst), "rb"))
    rng = np.random.default_rng(0)
    raw = []
    for i in rng.permutation(len(cases))[: a.n]:
        c = cases[i]
        h = [int(x) for x in c["history"]]
        s = int(c["pos_int"])
        bst, d, _ = search_cf(h, s, rules)
        if bst is not None and d == 0:
            raw.append({"h": h, "hh": bst[0], "s": s, "ori": c["ori_color"]})
    p_cf = pred([f["hh"] for f in raw])
    keep = [
        i
        for i, f in enumerate(raw)
        if float(p_cf[i, sorted(replay(f["hh"], rules).get_valid_moves())].sum())
        >= 0.99
    ]
    found = [raw[i] for i in keep]
    p_cf = p_cf[keep]
    bench = benchmark_from_cases(
        [{"history": f["h"], "pos_int": f["s"], "ori_color": f["ori"]} for f in found],
        **rules,
    )
    cur, tgt = case_targets(bench)
    tok, ln = oc.load(
        oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",), instance=inst)[
            "probe"
        ]
    )
    data = tokens_and_labels(
        [[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:20000], ln[:20000])],
        **rules,
    )
    grid = oa.fit_probe_grid(model, data, cache_dir=run_dir / "probes", log=None)
    lin = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(NP)}
    c_un = move_scorecard(
        pred([f["h"] for f in found]), bench.legal_pre, bench.legal_post
    )
    c_cf = move_scorecard(p_cf, bench.legal_pre, bench.legal_post)
    out = {
        "run": a.run,
        "instance": inst,
        "n_cases": len(found),
        "unedited": c_un["edit_index_union"],
        "ceiling_true_counterfactual": c_cf["edit_index_union"],
        "cf_mass_on_legal_post": float(
            np.mean([p_cf[i, bench.legal_post[i]].sum() for i in range(len(found))])
        ),
        "editors": {},
    }
    line = f"{a.run}: SAME {len(found)} clean cases | unedited {c_un['edit_index_union']:+.3f} | true counterfactual {c_cf['edit_index_union']:+.3f}"
    for ed, mode in (("PI", "pinv"), ("ND", "add_sub")):
        b = best[ed]
        pr, card = oa.linear_arm(
            model,
            bench,
            lin,
            tgt,
            cur,
            mode=mode,
            alpha=float(b["alpha"]),
            points={int(b["point"])},
        )
        out["editors"][ed] = {
            "point": b["point"],
            "alpha": b["alpha"],
            "edit_index_union": card["edit_index_union"],
            "mass_on_legal_post": float(
                np.mean([pr[i, bench.legal_post[i]].sum() for i in range(len(found))])
            ),
            "full_bench_edit_index_union": b["edit_index_union"],
        }
        line += f" | {ed}(pt{b['point']} α{b['alpha']}) {card['edit_index_union']:+.3f} (full bench {b['edit_index_union']:+.3f})"
    print(line, flush=True)
    (EXP / "scores" / f"honesty_check_{Path(a.run).name}.json").write_text(
        json.dumps(out, indent=1)
    )


if __name__ == "__main__":
    main()
