"""Othello data: vocabulary, synthetic games, board-state labels, activation harvest.

Ported 2026-08-31 from ``othello_transfer/othello_data.py`` with one structural change:
the generator and board rules come from the **vendored** copy
(``pim.environments.othello.vendor.othello``) instead of a ``sys.path`` reach into the
external clone. The retained conventions are unchanged and still matter:

Conventions copied from their ``train_probe_othello.py`` so numbers are comparable
-----------------------------------------------------------------------------------
* Model input is ``dix[:-1]`` — the first 59 tokens of a 60-move game.
* **Alignment**: activation at position ``t`` pairs with the board **after** move ``t``
  (their ``get_gt`` calls ``umpire(move)`` and *then* records). That board determines the
  legal moves the model is asked to predict at ``t``, which is what makes the probe
  target causally relevant.
* Board encoding is theirs: ``0 = white, 1 = blank, 2 = black`` (``get_state``).
* Positions past the end of a game are dropped, exactly as their ``valid_until`` does.

Deviations, both deliberate
---------------------------
* We call ``model.eval()``. Their probe script never does, so their harvest runs with
  dropout live at p=0.1 on the embedding, attention and residual paths.
* Games are **synthetic**, not championship. Their script hardcodes the championship
  data root even for the synthetic model, but that data is behind a dead link, and
  synthetic is the distribution the model was actually trained on.
"""

from __future__ import annotations

import multiprocessing
import random
from dataclasses import dataclass

import numpy as np
import torch

from pim.environments.othello.vendor.othello import OthelloBoardState, get_ood_game

# The four centre squares are occupied at t=0 and are never a legal move, so they carry no
# token and are padded back in when logits are laid out on the board.
CENTRE = (27, 28, 35, 36)
N_TILES = 64
N_CLASSES = 3  # white / blank / black, their `get_state` encoding
T_MODEL = 59  # the model sees the first 59 moves of a 60-move game


# ── vocabulary ────────────────────────────────────────────────────────────────


def canonical_vocab() -> dict[int, int]:
    """``{board square -> token}``, with the pad symbol -100 at token 0.

    Their ``CharDataset`` derives this from whatever games it is handed, so it is only
    stable if every one of the 60 non-centre squares appears. Building it explicitly
    makes the mapping independent of the sample.
    """
    squares = [s for s in range(N_TILES) if s not in CENTRE]
    return {-100: 0, **{sq: i + 1 for i, sq in enumerate(sorted(squares))}}


# ── synthetic games ───────────────────────────────────────────────────────────


def synthetic_games(n: int, seed: int = 0, n_workers: int | None = None) -> list[list[int]]:
    """``n`` games from THEIR generator, uniform over legal moves at every step.

    That uniformity is why ``uniform_over_legal`` in ``pim.metrics.othello_moves`` is the
    *true* conditional distribution of this data rather than an approximation of it.
    """
    n_workers = n_workers or multiprocessing.cpu_count()
    with multiprocessing.Pool(n_workers) as pool:
        return list(pool.imap(_one_game, [(i, seed) for i in range(n)], chunksize=64))


def _one_game(args) -> list[int]:
    """One game, seeded by its INDEX — deterministic given ``(seed, i)``.

    The obvious version seeds each worker process once, from its pid. That makes the
    corpus a function of process ids and chunk scheduling rather than of ``seed``, so
    ``seed=0`` produced a *different* 20k-game corpus on every run (three different row
    counts across three runs, 2026-08-20) and the probe cache could never hit. Seeding
    per work item fixes reproducibility and the cache together.
    """
    i, seed = args
    random.seed(seed * 1_000_003 + i)
    return get_ood_game(i)


# ── tokens and board-state labels ─────────────────────────────────────────────


@dataclass
class ProbeData:
    tokens: np.ndarray  # (N, 59) int64, pad = 0
    labels: np.ndarray  # (N, 59, 64) int8 — white 0 / blank 1 / black 2
    mine: np.ndarray  # (N, 59, 64) int8 — blank 0 / mine 1 / theirs 2
    mask: np.ndarray  # (N, 59) bool — True where the position is a real move
    lengths: np.ndarray  # (N,) int


def tokens_and_labels(games: list[list[int]]) -> ProbeData:
    """Tokenise and label, following their loop exactly (see module docstring)."""
    stoi = canonical_vocab()
    n, T = len(games), T_MODEL
    tokens = np.zeros((n, T), np.int64)
    labels = np.full((n, T, N_TILES), 1, np.int8)
    mine = np.zeros((n, T, N_TILES), np.int8)
    mask = np.zeros((n, T), bool)
    lengths = np.zeros(n, int)

    for i, g in enumerate(games):
        moves = g[:T]  # the model only ever sees the first 59
        lengths[i] = len(moves)
        tokens[i, : len(moves)] = [stoi[s] for s in moves]
        mask[i, : len(moves)] = True
        board = OthelloBoardState()
        for t, mv in enumerate(moves):
            board.umpire(mv)
            st = (board.state + 1).flatten().astype(np.int8)  # white 0 / blank 1 / black 2
            labels[i, t] = st
            # "mine" = the player about to move at this position; `next_hand_color` is
            # +1 for black, -1 for white, and it is NOT parity because of passes.
            nxt = 2 if board.next_hand_color > 0 else 0
            mine[i, t] = np.where(st == 1, 0, np.where(st == nxt, 1, 2))
    return ProbeData(tokens, labels, mine, mask, lengths)


def flatten_rows(data: ProbeData, target: str = "state") -> tuple[np.ndarray, np.ndarray]:
    """(row_index_of_sequence, positions) → the flat (activation, label) row layout."""
    y = data.labels if target == "state" else data.mine
    seq_idx = np.repeat(np.arange(len(data.tokens))[:, None], data.tokens.shape[1], 1)
    return seq_idx[data.mask], y[data.mask]


# ── activation harvest ────────────────────────────────────────────────────────


@torch.no_grad()
def harvest_point(model, tokens: np.ndarray, point: int, batch: int = 512) -> np.ndarray:
    """(N, 59, d_model) residual stream at one residual point.

    One residual point at a time: all nine at 20k games would be ~22 GB, one is ~2.4 GB.
    """
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(tokens), batch):
        idx = torch.from_numpy(tokens[i : i + batch]).to(dev)
        rs = model.residual_stack(idx)
        out.append(rs[point].float().cpu().numpy())
    return np.concatenate(out, 0)


# ── logits → board ────────────────────────────────────────────────────────────


def board_probs(logits: torch.Tensor) -> np.ndarray:
    """(B, 61) next-move logits → (B, 64) probability laid out on the board.

    Their mapping: drop the pad logit, softmax over the 60 move tokens, then pad zeros
    back into the four centre squares.
    """
    p = torch.softmax(logits[:, 1:], dim=-1)
    pad = torch.zeros(len(p), 2, device=p.device, dtype=p.dtype)
    out = torch.cat([p[:, :27], pad, p[:, 27:33], pad, p[:, 33:]], dim=1)
    return out.float().cpu().numpy()
