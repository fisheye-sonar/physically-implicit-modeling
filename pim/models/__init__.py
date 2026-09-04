"""pim.models — the canonical architectures, each swappable between the two tasks.

Two architectures × two task heads:

    transformer_s.py   Transformer-S (~3.2M, ours): banded-causal attention, RoPE,
                       pre-norm blocks. `TransformerS` (regression) /
                       `TransformerSTokens` (move classification).
    transformer_l.py   Transformer-L (~25M, Li et al.'s minGPT, vendored): full
                       causal attention, learned absolute positions.
                       `TransformerL` (regression) / `TransformerLTokens` (tokens).

    protocol.py        THE surface every model implements — read this first.
    registry.py        explicit name → builder + the one checkpoint loader.
    blocks.py          shared attention pieces (RoPE, band mask, self-attention).

Cross-environment comparisons are only meaningful because the architecture is
*identical* across environments up to the input/output projection and the loss —
that is the invariant this package exists to protect.
"""

from pim.models.protocol import WorldModel, n_points
from pim.models.registry import BUILDERS, CheckpointInfo, build, load_checkpoint, load_run
from pim.models.transformer_l import ArchState, TransformerL, TransformerLTokens
from pim.models.transformer_s import (
    ModelConfig,
    TransformerS,
    TransformerSTokens,
    TransformerState,
)

__all__ = [
    "WorldModel",
    "n_points",
    "BUILDERS",
    "CheckpointInfo",
    "build",
    "load_checkpoint",
    "load_run",
    "ModelConfig",
    "TransformerS",
    "TransformerSTokens",
    "TransformerState",
    "ArchState",
    "TransformerL",
    "TransformerLTokens",
]
from pim.models.recurrent import RecurrentConfig, RecurrentL  # noqa: E402,F401
