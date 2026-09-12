#!/usr/bin/env python
"""Tokenise a noiseless discworld instance: frames → token ids (pim.environments.discworld.tokens).

    python scripts/make_discworld_tokens.py --instance dw-8ray

Writes datasets/discworld/<instance>/tokens/ (train.i16, test/edits.npy, vocab.npz,
meta.json); the source splits are resolved through pim.environments.layout (v2). Additive;
safe to re-run (overwrites the tokens/ files only).
Run under a memory cap like every corpus job: peak ≈ one chunk of frames (~0.4 GB).
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments.discworld.tokens import tokenize_instance  # noqa: E402
from pim.environments.layout import instance_root  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--instance", default="dw-8ray")
    p.add_argument("--chunk", type=int, default=250_000)
    a = p.parse_args()
    tokenize_instance(instance_root("discworld", a.instance), chunk=a.chunk)


if __name__ == "__main__":
    main()
