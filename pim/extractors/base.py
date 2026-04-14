"""Base types for extractors.

StateDefinition describes what physical quantity to recover from model
internal states. It is intentionally general — state_shape can represent
object-centric quantities (e.g. (n_objects, 2) for 2D positions) or global
environment attributes (e.g. (4,) for a gravity vector).

Extractor is a structural Protocol: any nn.Module with the right forward
signature qualifies, no inheritance needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch


@dataclass
class StateDefinition:
    """Describes one physical quantity to recover from model internal states.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "positions", "velocities", "gravity".
    state_shape : tuple[int, ...]
        Per-timestep shape of the target state. Examples:
            (3, 2)  — 2D positions for 3 objects
            (3,)    — scalar per object (e.g. reflectivity)
            (4,)    — global environment attribute (non-object-centric)
    extract_fn : callable
        Maps a batch dict (from DataLoader) to target tensor of shape
        (B, T, *state_shape). Allows arbitrary transformations.

    Examples
    --------
    >>> POSITIONS_2D = StateDefinition(
    ...     name="positions",
    ...     state_shape=(3, 2),
    ...     extract_fn=lambda batch: batch["positions"],
    ... )
    >>> VELOCITIES_2D = StateDefinition(
    ...     name="velocities",
    ...     state_shape=(3, 2),
    ...     extract_fn=lambda batch: batch["velocities"],
    ... )
    """

    name: str
    state_shape: tuple[int, ...]
    extract_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor]

    @property
    def output_dim(self) -> int:
        """Flattened output dimensionality (= product of state_shape)."""
        return math.prod(self.state_shape)


class Extractor(Protocol):
    """Structural protocol for any extractor module.

    Any nn.Module whose forward maps (..., hidden_size) → (..., *state_shape)
    satisfies this protocol.
    """

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Map hidden states to decoded env state.

        Parameters
        ----------
        h : (..., hidden_size)

        Returns
        -------
        decoded : (..., *state_shape)
        """
        ...
