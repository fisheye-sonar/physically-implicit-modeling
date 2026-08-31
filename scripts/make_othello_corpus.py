#!/usr/bin/env python3
"""Generate (or extend) the canonical Othello corpus — thin entry over
``pim.environments.othello.corpus``.

    python scripts/make_othello_corpus.py                # the full 20M train pool
    python scripts/make_othello_corpus.py 1000000        # a smaller pool (still a prefix)
    python scripts/make_othello_corpus.py 20000000 test,probe

Splits land in ``datasets/othello/oth-uniform/corpus/`` and are disjoint index ranges of
one seed (hash-verified). See the instance manifest for the split law.
"""

import runpy
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

runpy.run_module("pim.environments.othello.corpus", run_name="__main__")
