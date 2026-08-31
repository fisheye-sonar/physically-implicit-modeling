"""pim.training — the ONE canonical training loop and its data sources.

    train.py    TrainConfig (the matched BIG20M recipe), DataSource, the loop,
                the two objectives (MSE next-obs / padded-CE next-move).
    sources.py  how each environment feeds the loop (discworld memmap stream,
                Othello on-GPU tokens).
    stream.py   BlockStream, the shuffled-block memmap reader.

Entry point: ``scripts/train.py``. Every run writes config.json + commit_sha +
metrics.jsonl + arch-stamped checkpoints into its own ``runs/<topic>/<name>/`` dir.
"""

from pim.training.sources import discworld_source, othello_source
from pim.training.stream import BlockStream
from pim.training.train import (
    DataSource,
    TrainConfig,
    ce_next_move,
    mse_next_obs,
    train,
    xy_tokens,
)

__all__ = [
    "TrainConfig",
    "DataSource",
    "train",
    "mse_next_obs",
    "ce_next_move",
    "xy_tokens",
    "BlockStream",
    "discworld_source",
    "othello_source",
]
