#!/usr/bin/env python
"""Synthesise an Othello instance's 1001 intervention cases from its held-out TEST split.

    python scripts/make_othello_edits.py --instance oth-noflip

Li et al.'s shipped benchmark only fits oth-uniform (its cases are real flip-Othello
positions). Any other instance gets a set built by the same recipe
(`pim.environments.othello.bench.synthesise_cases`), with the shipped set's prefix-length
distribution, written to datasets/othello/<instance>/edits/cases_1001.pkl + a manifest.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments import layout  # noqa: E402
from pim.environments.othello import corpus as oc  # noqa: E402
from pim.environments.othello.bench import shipped_length_distribution, synthesise_cases  # noqa: E402
from pim.environments.othello.data import canonical_vocab  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--length", type=int, default=20,
                    help="the FIXED prefix length every case is cut at (2026-09-12: 20 moves — the "
                         "discworld bench edits at a fixed frame too; 0 = Li's 5-30 length mix)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    # the instance's own EDITS games — an index range disjoint from train, the gates' test
    # split and both probe corpora (corpus.EDITS_LO; generated here if absent, seconds)
    tok, ln = oc.load(oc.build(only=("edits",), instance=a.instance, log=lambda s: None)["edits"])
    itos = {v: k for k, v in canonical_vocab().items()}
    hist = [[int(itos[int(t)]) for t in row[:L]] for row, L in zip(tok, ln)]
    lengths = {a.length: a.n} if a.length else shipped_length_distribution()
    cases, manifest = synthesise_cases(hist, a.n, lengths, seed=a.seed, **oc.rules_of(a.instance))
    layout.ensure_marker("othello", a.instance)              # a new instance is born in layout v2
    out = layout.edits_dir("othello", a.instance, "v1")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"cases_{a.n}.pkl", "wb") as f:
        pickle.dump(cases, f)
    manifest.update({"instance": a.instance, "source": f"{a.instance} EDITS split ({len(hist)} games, "
                     f"index range [{oc.EDITS_LO}, {oc.EDITS_LO + oc.EDITS_N}))",
                     "prefix_length": a.length or "Li's 5-30 mix",
                     "recipe": "one occupied non-centre tile recoloured; rejected if the legal set is "
                               "unchanged or empty (bench.synthesise_cases)",
                     "minutes": round((time.time() - t0) / 60, 1)})
    (out / f"cases_{a.n}.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(cases)} cases -> {out / f'cases_{a.n}.pkl'}  [{manifest['minutes']} min]")


if __name__ == "__main__":
    main()
