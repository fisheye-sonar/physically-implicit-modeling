#!/usr/bin/env python
"""Is the MSE-on-one-hot head actually regressing a distribution? And what changes if we
make it one?

For an Othello token model trained with `mse_next_move_onehot` (`output_kind="raw"`), on
the held-out TEST split:
  1. raw-output statistics per position — sum over the 60 move outputs, negative mass,
     min / max, L1 distance to its clip-and-renormalised version, share of positions whose
     argmax is legal;
  2. the canonical gates under each output kind — raw (as the scorer reads it) and
     clipnorm (clip at 0, renormalise): legal mass, top-1 legality, CE vs the Bayes floor;
  3. the run's canonical best arms (PI, ND, GS from scores.json) re-scored under clipnorm
     beside their raw values — the Edit Index / fidelity under "made a distribution".
Everything is the canonical code path (`data.move_probs`, `arms.gates`, `arms.linear_arm`,
`arms.grad_steer_arm`, cached probes — a cache miss aborts); the only variable is the
model's `output_kind` attribute, flipped in memory for the duration of each pass.

Outputs: experiments/othello_mse_head/scores/distribution_check_<run>.json + summary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pim.environments.othello import arms as oa  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import case_targets, load_benchmark  # noqa: E402
from pim.environments.othello.data import (  # noqa: E402
    canonical_vocab, move_probs, tokens_and_labels)
from pim.metrics.othello_moves import move_fidelity_ratio  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "othello_mse_head"
BLOCK = 59


def probe_games(n: int):
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])])


def cached_grid(model, data, cache_dir: Path, dev: str):
    store = ProbeCache(cache_dir)
    fname, prov = store.key(
        model, kind="othello_grid", targets=["mine"], families=["linear", "mlp"],
        splits=["sequence"], holdout=0.2, epochs=200, batch=4096, lr=1e-3, seed=0,
        n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=model.n_layers + 1)
    blob = store.load(fname, prov, device=dev)
    if blob is None:
        sys.exit(f"no cached canonical probe grid under {cache_dir} — refusing to refit")
    return blob["probes"]


@torch.no_grad()
def raw_stats(model, tokens, lengths, legal, dev, batch=512) -> dict:
    stoi = canonical_vocab()
    s = {"sum": [], "neg_mass": [], "min": [], "max": [], "l1_to_clipnorm": [],
         "argmax_legal": 0, "n": 0}
    for i in range(0, len(tokens), batch):
        tk = torch.from_numpy(tokens[i : i + batch]).long().to(dev)
        out = model.logits(tk[:, :BLOCK])
        raw = move_probs(out, "raw")
        cn = move_probs(out, "clipnorm")
        for r in range(len(tk)):
            L = int(lengths[i + r])
            for t in range(L - 1):
                lm = legal[i + r][t]
                if not lm:
                    continue
                v = raw[r, t]
                s["sum"].append(float(v.sum()))
                s["neg_mass"].append(float(-v[v < 0].sum()))
                s["min"].append(float(v.min()))
                s["max"].append(float(v.max()))
                s["l1_to_clipnorm"].append(float((v - cn[r, t]).abs().sum()))
                s["argmax_legal"] += int((int(v.argmax()) + 1) in {stoi[q] for q in lm})
                s["n"] += 1
    out = {k: {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)),
               "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
           for k, v in s.items() if isinstance(v, list)}
    out["argmax_legal"] = s["argmax_legal"] / max(s["n"], 1)
    out["n_positions"] = s["n"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/objective_ablation/L-oth-20m-mse")
    ap.add_argument("--n-games", type=int, default=10_000)
    a = ap.parse_args()
    t0 = time.time()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    native = getattr(model, "output_kind", "logits")
    S = json.loads((run_dir / "scores.json").read_text())
    (EXP / "scores").mkdir(parents=True, exist_ok=True)

    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s_: None, only=("test",))["test"])
    tok, ln = tok[: a.n_games], ln[: a.n_games]
    legal = oa.legal_sets(tok, ln)
    res = {"run": run_dir.name, "arch": info.arch, "native_output_kind": native,
           "n_games": int(len(tok)), "raw_stats": raw_stats(model, tok, ln, legal, dev),
           "gates": {}, "arms": {}}
    print("raw stats:", json.dumps({k: (round(v["mean"], 4) if isinstance(v, dict) else v)
                                    for k, v in res["raw_stats"].items()}), flush=True)
    for kind in ("raw", "clipnorm"):
        model.output_kind = kind
        g = oa.gates(model, tok, ln, log=None)
        res["gates"][kind] = g
        print(f"gates[{kind}]: legal_mass {g['legal_mass']:.4f} top1_legal {g['top1_legal']:.4f} "
              f"ce {g['ce']:.4f} (bayes {g['bayes_ce']:.4f}) sum {g['out_sum_mean']:.3f} "
              f"neg {g['out_neg_mass_mean']:.4f}", flush=True)

    # the canonical best arms, re-read under each output kind
    bench = load_benchmark()
    cur, tgt = case_targets(bench)
    probes = cached_grid(model, probe_games(S["settings"]["oth_probe_games"]),
                         run_dir / "probes", dev)
    npnt = n_points(model)
    lin = {p: probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
    mlp = {p: probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}
    for kind in ("raw", "clipnorm"):
        model.output_kind = kind
        uns = oa.unsteered_probs(model, bench)
        u = oa.unsteered(model, bench)
        row = {"unedited": u["edit_index_union"]}
        for ed, b in S["best"].items():
            if not b:
                continue
            if ed in ("PI", "ND"):
                pr, card = oa.linear_arm(model, bench, lin, tgt, cur,
                                         mode="pinv" if ed == "PI" else "add_sub",
                                         alpha=b["alpha"], points={b["point"]})
            else:
                pr, card = oa.grad_steer_arm(model, bench, mlp, b["point"], alpha=b["alpha"],
                                             n_steps=S["settings"]["oth_gs_steps"],
                                             beta=S["settings"]["oth_gs_beta"], target_labels=tgt)
            row[ed] = {"arm": f"pt{b['point']}·α{b['alpha']:g}",
                       "edit_index": card["edit_index_union"],
                       "fidelity_ratio": move_fidelity_ratio(pr, uns, bench.legal_post),
                       "li_error_vs_post": card["li_error_vs_post"],
                       "legal_mass": card["legal_mass"]}
        res["arms"][kind] = row
        print(f"arms[{kind}]: unedited {row['unedited']:+.3f}  " + "  ".join(
            f"{ed} {r['edit_index']:+.3f}/{r['fidelity_ratio']:.2f}" for ed, r in row.items()
            if isinstance(r, dict)), flush=True)
    model.output_kind = native
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out = EXP / "scores" / f"distribution_check_{run_dir.name}.json"
    out.write_text(json.dumps(res, indent=1, default=float))

    rs, G, A = res["raw_stats"], res["gates"], res["arms"]
    md = [f"# {run_dir.name} — is the MSE head a distribution?", "",
          f"native output kind `{native}`; {res['n_games']:,} held-out test games, "
          f"{rs['n_positions']:,} positions", "",
          "| raw output over the 60 move outputs | mean | p05 | p50 | p95 |", "|---|---|---|---|---|"]
    for k in ("sum", "neg_mass", "min", "max", "l1_to_clipnorm"):
        v = rs[k]
        md.append(f"| {k} | {v['mean']:.4f} | {v['p05']:.4f} | {v['p50']:.4f} | {v['p95']:.4f} |")
    md += ["", f"argmax legal: {rs['argmax_legal']:.4f}", "",
           "| gates | raw | clipnorm | Bayes / CE-model reference |", "|---|---|---|---|",
           f"| legal mass | {G['raw']['legal_mass']:.4f} | {G['clipnorm']['legal_mass']:.4f} | 1.0 / 0.93 |",
           f"| top-1 legal | {G['raw']['top1_legal']:.4f} | {G['clipnorm']['top1_legal']:.4f} | 1.0 / 0.987 |",
           f"| CE | {G['raw']['ce']:.4f} | {G['clipnorm']['ce']:.4f} | {G['raw']['bayes_ce']:.4f} (Bayes) |",
           "", "| editability (canonical best arms) | raw | clipnorm |", "|---|---|---|",
           f"| unedited | {A['raw']['unedited']:+.3f} | {A['clipnorm']['unedited']:+.3f} |"]
    for ed in ("PI", "ND", "GS"):
        if ed in A["raw"]:
            md.append(f"| {ed} {A['raw'][ed]['arm']} | {A['raw'][ed]['edit_index']:+.3f} / "
                      f"{A['raw'][ed]['fidelity_ratio']:.2f} | {A['clipnorm'][ed]['edit_index']:+.3f} / "
                      f"{A['clipnorm'][ed]['fidelity_ratio']:.2f} |")
    (EXP / "scores" / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"done  {out}  [{res['minutes']} min]")


if __name__ == "__main__":
    main()
