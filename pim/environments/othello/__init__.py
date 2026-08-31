"""pim.environments.othello — Li et al.'s synthetic Othello environment.

Next-token prediction over move sequences; the generator draws uniformly from the legal
set, so uniform-over-legal is the Bayes-optimal predictor (the anchor every Othello
metric calibrates against).

Module map:

    vendor/     byte-identical third-party code (MIT, Kenneth Li): board rules +
                game generator (`othello.py`), their minGPT (`mingpt_model.py`),
                and the shipped 1001-case intervention benchmark
    data.py     vocabulary, synthetic games (index-seeded), board-state labels,
                activation harvest, logits→board mapping
    corpus.py   the nested scale ladder (.npz token caches, disjointness asserts)
    bench.py    the editability bench: Benchmark buckets + mine/theirs case targets
"""

from pim.environments.othello.bench import Benchmark, case_targets, load_benchmark
from pim.environments.othello.data import (
    CENTRE,
    N_CLASSES,
    N_TILES,
    ProbeData,
    board_probs,
    canonical_vocab,
    flatten_rows,
    harvest_point,
    synthetic_games,
    tokens_and_labels,
)

__all__ = [
    "CENTRE",
    "N_CLASSES",
    "N_TILES",
    "ProbeData",
    "board_probs",
    "canonical_vocab",
    "flatten_rows",
    "harvest_point",
    "synthetic_games",
    "tokens_and_labels",
    "Benchmark",
    "case_targets",
    "load_benchmark",
]
