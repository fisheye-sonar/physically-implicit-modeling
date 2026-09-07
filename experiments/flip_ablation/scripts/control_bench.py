#!/usr/bin/env python
"""The like-for-like control: score L-oth-20m on a SYNTHESISED flip bench of the noflip recipe.

`L-oth-noflip-20m` is scored on 1001 cases synthesised from its test split with Li's recipe
and prefix-length mix (`bench.synthesise_cases`), because Li's shipped cases are flip-Othello
positions. `L-oth-20m` is canonically scored on the shipped 1001. To compare the two runs on
benches of IDENTICAL construction, this synthesises 1001 FLIP cases the same way (from the
oth-uniform test split) and scores L-oth-20m's unedited output and its canonical best arms
(PI / ND / GS from scores.json) on them, with the run's cached probes (a cache miss aborts).

Output: experiments/flip_ablation/scores/control_L-oth-20m_synth_flip_bench.json + summary.md.
The synthesised flip cases are kept in experiments/flip_ablation/cases/ (they are a control,
not the instance's canonical bench).
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import (benchmark_from_cases, case_targets,  # noqa: E402
                                            shipped_length_distribution, synthesise_cases)
from pim.environments.othello.data import canonical_vocab, tokens_and_labels  # noqa: E402
from pim.metrics.othello_moves import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "flip_ablation"
RUN = REPO / "runs" / "initial_othello_comparison" / "L-oth-20m"
DEV = "cuda"


def cached_grid(model, data, cache_dir: Path):
    store = ProbeCache(cache_dir)
    fname, prov = store.key(model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"],
                            splits=["sequence"], holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0,
                            n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()), n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=DEV)
    if blob is None:
        sys.exit(f"no cached canonical probe grid under {cache_dir}")
    return blob["probes"]


def main() -> None:
    t0 = time.time()
    (EXP / "cases").mkdir(exist_ok=True)
    (EXP / "scores").mkdir(exist_ok=True)
    tok, ln = oc.load(oc.build(only=("test",), instance="oth-uniform", log=lambda s: None)["test"])
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    cases, man = synthesise_cases(hist, 1001, shipped_length_distribution(), seed=0, flip=True, log=None)
    with open(EXP / "cases" / "synth_flip_1001.pkl", "wb") as f:
        pickle.dump(cases, f)
    bench = benchmark_from_cases(cases, flip=True)
    S = json.loads((RUN / "scores.json").read_text())
    model, info = load_checkpoint(RUN / "best_model.pt", device=DEV)
    model.eval()
    # the run's own probes: the same games master_eval used
    ptok, pln = oc.load(oc.build(only=("probe",), instance="oth-uniform", log=lambda s: None)["probe"])
    n = S["settings"]["oth_probe_games"]
    data = tokens_and_labels([[itos[int(t)] for t in row[:L]] for row, L in zip(ptok[:n], pln[:n])])
    probes = cached_grid(model, data, RUN / "probes")
    npnt = n_points(model)
    lin = {p: probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
    mlp = {p: probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}
    cur, tgt = case_targets(bench)
    uns = oa.unsteered_probs(model, bench)
    u = oa.unsteered(model, bench)
    out = {"run": "L-oth-20m", "bench": "synthesised flip cases (Li recipe + shipped length mix, seed 0)",
           "n_cases": len(cases), "manifest": man,
           "unedited": {k: v for k, v in u.items() if isinstance(v, (int, float))}, "best_arms": {}}
    for ed, b in S["best"].items():
        if not b:
            continue
        if ed in ("PI", "ND"):
            pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode="pinv" if ed == "PI" else "add_sub",
                                     alpha=b["alpha"], points={b["point"]})
        else:
            pr, card = oa.grad_steer_arm(model, bench, mlp, b["point"], alpha=b["alpha"],
                                         n_steps=S["settings"]["oth_gs_steps"], beta=S["settings"]["oth_gs_beta"],
                                         target_labels=tgt)
        out["best_arms"][ed] = {"arm": f"pt{b['point']}·α{b['alpha']:g}",
                                "edit_index_union": card["edit_index_union"],
                                "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                                "li_error_vs_post": card["li_error_vs_post"], "legal_mass": card["legal_mass"],
                                "shipped_bench_edit_index_union": b["edit_index_union"],
                                "shipped_bench_fidelity_ratio": b["fidelity_ratio"]}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    (EXP / "scores" / "control_L-oth-20m_synth_flip_bench.json").write_text(json.dumps(out, indent=1, default=float))
    md = ["# Control: L-oth-20m on a synthesised FLIP bench (same recipe as oth-noflip's cases)", "",
          f"1001 cases from the oth-uniform TEST split; unedited EI {out['unedited']['edit_index_union']:+.3f} "
          f"(shipped bench {S['unedited']['edit_index_union']:+.3f})", "",
          "| editor · canonical best arm | synthesised flip bench EI / fid | shipped Li bench EI / fid |", "|---|---|---|"]
    for ed, r in out["best_arms"].items():
        md.append(f"| {ed} {r['arm']} | {r['edit_index_union']:+.3f} / {r['fidelity_ratio']:.2f} | "
                  f"{r['shipped_bench_edit_index_union']:+.3f} / {r['shipped_bench_fidelity_ratio']:.2f} |")
    (EXP / "scores" / "summary_control.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("done", out["minutes"], "min")


if __name__ == "__main__":
    main()
