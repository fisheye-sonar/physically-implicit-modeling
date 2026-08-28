"""Data, harvest and metrics for `othello_transfer` — Li et al.'s world, our code.

Everything here is *data preparation and scoring*. The probe, the write and the descent
all live in `../othello_gpt/othello_probe.py` and are used unmodified; the only thing this
module knows about the intervention is how to score its output.

Conventions copied from `train_probe_othello.py` so the numbers are comparable
---------------------------------------------------------------------------
* Model input is ``dix[:-1]`` — the first 59 tokens of a 60-move game.
* **Alignment**: activation at position ``t`` pairs with the board **after** move ``t``
  (their ``get_gt`` calls ``umpire(move)`` and *then* records). That board is what
  determines the legal moves the model is asked to predict at ``t``, which is the whole
  reason the probe target is causally relevant.
* Board encoding is theirs: ``0 = white, 1 = blank, 2 = black`` (``get_state``).
* Positions past the end of a game are dropped, exactly as their ``valid_until`` does.

Deviations, both deliberate
---------------------------
* We call ``model.eval()``. Their probe script never does, so their harvest runs with
  dropout live at p=0.1 on the embedding, attention and residual paths.
* Games are **synthetic**, not championship. Their script hardcodes
  ``data_root="data/othello_championship"`` even for the synthetic model, but that data is
  behind a dead link, and synthetic is the distribution this model was actually trained on.
"""

from __future__ import annotations

import multiprocessing
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

OTHELLO_ROOT = Path("/home/sevan/research/PIM/othello_world")

# The four centre squares are occupied at t=0 and are never a legal move, so they carry no
# token and are padded back in when logits are laid out on the board.
CENTRE = (27, 28, 35, 36)
N_TILES = 64
N_CLASSES = 3  # white / blank / black, their `get_state` encoding


# ── vocabulary ────────────────────────────────────────────────────────────────


def canonical_vocab() -> dict[int, int]:
    """``{board square -> token}``, with the pad symbol -100 at token 0.

    `CharDataset` derives this from whatever games it is handed, so it is only stable if
    every one of the 60 non-centre squares appears. Building it explicitly makes the
    mapping independent of the sample; `assert_vocab_matches` checks their construction
    agrees.
    """
    squares = [s for s in range(N_TILES) if s not in CENTRE]
    return {-100: 0, **{sq: i + 1 for i, sq in enumerate(sorted(squares))}}


def assert_vocab_matches(char_dataset) -> None:
    ours = canonical_vocab()
    theirs = char_dataset.stoi
    if ours != theirs:
        raise AssertionError(
            "CharDataset vocabulary differs from the canonical mapping — the sampled "
            "games do not cover all 60 move squares, so tokens would not match the "
            "checkpoint's training vocabulary."
        )
    if char_dataset.block_size != 59:
        raise AssertionError(
            f"block_size {char_dataset.block_size} != 59; the longest sampled game is "
            "shorter than 60 moves, which shifts every position embedding."
        )


# ── synthetic games ───────────────────────────────────────────────────────────


def synthetic_games(n: int, seed: int = 0, n_workers: int | None = None) -> list[list[int]]:
    """`n` games from THEIR generator, `data.othello.get_ood_game`.

    Called directly rather than through `get_othello`, which for ``ood_num > 1000``
    writes a pickle into a directory their `.gitignore` expects but does not create.
    That generator picks ``random.choice(possible_next_steps)`` — uniform over legal
    moves — which is why the uniform-over-legal reference in `edit_index` is the true
    conditional distribution of this data rather than an approximation of it.
    """
    n_workers = n_workers or multiprocessing.cpu_count()
    with multiprocessing.Pool(n_workers) as pool:
        return list(pool.imap(_one_game, [(i, seed) for i in range(n)], chunksize=64))


def _one_game(args) -> list[int]:
    """One game, seeded by its INDEX — deterministic given ``(seed, i)``.

    The obvious version seeds each worker process once, from its pid. That makes the corpus
    a function of process ids and chunk scheduling rather than of ``seed``, so ``SEED = 0``
    produced a *different* 20k-game corpus on every run (1,179,692 / 1,179,508 / 1,179,665
    rows across three runs, 2026-08-20) and the probe cache — keyed on row count — could
    never hit. Seeding per work item fixes the reproducibility and the cache together.
    """
    import random
    import sys

    i, seed = args
    if str(OTHELLO_ROOT) not in sys.path:
        sys.path.insert(0, str(OTHELLO_ROOT))
    from data.othello import get_ood_game

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
    import sys

    if str(OTHELLO_ROOT) not in sys.path:
        sys.path.insert(0, str(OTHELLO_ROOT))
    from data.othello import OthelloBoardState

    stoi = canonical_vocab()
    n, T = len(games), 59
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
def harvest_point(shim, tokens: np.ndarray, point: int, batch: int = 512) -> np.ndarray:
    """(N, 59, d_model) residual stream at one residual point.

    One residual point at a time: all nine at 20k games would be ~22 GB, one is ~2.4 GB.
    """
    dev = next(shim.parameters()).device
    out = []
    for i in range(0, len(tokens), batch):
        idx = torch.from_numpy(tokens[i : i + batch]).to(dev)
        rs = shim.residual_stack(idx)
        out.append(rs[point].float().cpu().numpy())
    return np.concatenate(out, 0)


# ── the intervention benchmark ────────────────────────────────────────────────


@dataclass
class Benchmark:
    tokens: list[np.ndarray]  # per bucket, (B, L) int64
    case_ids: list[np.ndarray]  # per bucket, indices into the 1001
    pos_int: np.ndarray  # (1001,) intervened square
    new_class: np.ndarray  # (1001,) requested class, their `2 - ori_color`
    legal_pre: list[list[int]]
    legal_post: list[list[int]]

    @property
    def n_cases(self) -> int:
        return len(self.pos_int)


def load_benchmark() -> Benchmark:
    """Li's shipped 1001 cases, grouped into equal-length buckets.

    Buckets exist for one reason: `make_intervention_hook` writes ``x[:, -1]``, so every
    row in a batch must have its last real move at the same index. Grouping by history
    length keeps that true without padding and without touching the hook.
    """
    import sys

    if str(OTHELLO_ROOT) not in sys.path:
        sys.path.insert(0, str(OTHELLO_ROOT))
    from data.othello import OthelloBoardState

    with open(OTHELLO_ROOT / "intervention_benchmark.pkl", "rb") as f:
        ds = pickle.load(f)
    stoi = canonical_vocab()

    pos_int = np.array([c["pos_int"] for c in ds], int)
    new_class = np.array([int(2 - c["ori_color"]) for c in ds], int)
    legal_pre, legal_post = [], []
    for c, sq, new in zip(ds, pos_int, new_class):
        pre = OthelloBoardState()
        pre.update(c["history"], prt=False)
        legal_pre.append(sorted(pre.get_valid_moves()))
        post = OthelloBoardState()
        post.update(c["history"], prt=False)
        post.state[sq // 8, sq % 8] = new - 1
        legal_post.append(sorted(post.get_valid_moves()))

    by_len: dict[int, list[int]] = {}
    for i, c in enumerate(ds):
        by_len.setdefault(len(c["history"]), []).append(i)
    toks, ids = [], []
    for L in sorted(by_len):
        members = np.array(by_len[L], int)
        ids.append(members)
        toks.append(np.array([[stoi[s] for s in ds[i]["history"]] for i in members], np.int64))
    return Benchmark(toks, ids, pos_int, new_class, legal_pre, legal_post)


# ── metrics ───────────────────────────────────────────────────────────────────


def board_probs(logits: torch.Tensor) -> np.ndarray:
    """(B, 61) next-move logits → (B, 64) probability laid out on the board.

    Their mapping: drop the pad logit, softmax over the 60 move tokens, then pad zeros
    back into the four centre squares.
    """
    p = torch.softmax(logits[:, 1:], dim=-1)
    pad = torch.zeros(len(p), 2, device=p.device, dtype=p.dtype)
    out = torch.cat([p[:, :27], pad, p[:, 27:33], pad, p[:, 33:]], dim=1)
    return out.float().cpu().numpy()


def li_error(probs: np.ndarray, legal: list[list[int]]) -> np.ndarray:
    """Their §4.2 metric: top-N predictions vs a legal-move set, false pos + false neg.

    ``N = len(legal)``, so both sets have the same size and the error is
    ``2 x (N - overlap)``. Lower is better; their null-intervention baseline on the
    natural benchmark is 2.68 and their best intervention is 0.12.
    """
    out = np.full(len(probs), np.nan)
    for i, L in enumerate(legal):
        if not L:
            continue
        top = set(np.argsort(-probs[i])[: len(L)].tolist())
        out[i] = 2 * (len(L) - len(top & set(L)))
    return out


def uniform_over_legal(legal: list[int]) -> np.ndarray:
    v = np.zeros(N_TILES, np.float32)
    if legal:
        v[list(legal)] = 1.0 / len(legal)
    return v


def edit_index(
    probs: np.ndarray,
    legal_pre: list[list[int]],
    legal_post: list[list[int]],
    support: str = "union",
) -> np.ndarray:
    """This thread's translation of the repo's Edit Index onto next-move distributions.

    Same formula as `scripts.editability_metrics`: ``(d_uned - d_edit)/(d_uned + d_edit)``
    with ``d_.`` an RMSE against a ground-truth world, scored on the squares where the two
    worlds differ. **+1** = the output is the edited world, **-1** = the unedited one.

    The reference world is the **uniform distribution over legal moves**, which is not an
    approximation here: their synthetic generator draws moves uniformly from the legal set,
    so uniform-over-legal is the true conditional distribution and the Bayes-optimal
    predictor. Measured 2026-08-20, the unedited model sits 0.0016 RMSE from it per square
    against a 0.0193 separation between the two worlds — a 12x margin.

    ``support="union"`` is the faithful translation of "rays where the two ground-truth
    worlds differ", because the two uniform references renormalise (1/|L0| vs 1/|L1|) and
    so differ on *shared* legal squares too, in 69.9% of cases. ``support="symdiff"``
    scores only squares whose legality changed — a narrower question, reported alongside
    but never quoted as the same quantity. Floors for the unedited model, all 1001 cases:
    **-0.829 (union)**, -0.943 (symdiff); a perfect predictor of the unedited world scores
    exactly -1 on both.
    """
    out = np.full(len(probs), np.nan)
    for i, (L0, L1) in enumerate(zip(legal_pre, legal_post)):
        s0, s1 = set(L0), set(L1)
        idx = np.array(sorted(s0 | s1 if support == "union" else s0 ^ s1), int)
        if idx.size == 0:
            continue
        g0, g1 = uniform_over_legal(L0), uniform_over_legal(L1)
        d_un = float(np.sqrt(((probs[i, idx] - g0[idx]) ** 2).mean()))
        d_ed = float(np.sqrt(((probs[i, idx] - g1[idx]) ** 2).mean()))
        if d_un + d_ed == 0:
            continue
        out[i] = (d_un - d_ed) / (d_un + d_ed)
    return out


def scorecard(probs: np.ndarray, bench: Benchmark) -> dict:
    """Every number an arm reports, in one place.

    `li_error_vs_post` is the headline (their metric). `li_error_vs_pre` is the guard:
    a null intervention is low on `pre` and high on `post`, a successful one is the
    reverse, and an arm that **degraded** the model is high on both — which their metric
    alone cannot distinguish.
    """
    e_post = li_error(probs, bench.legal_post)
    e_pre = li_error(probs, bench.legal_pre)
    ei_u = edit_index(probs, bench.legal_pre, bench.legal_post, "union")
    ei_s = edit_index(probs, bench.legal_pre, bench.legal_post, "symdiff")
    return {
        "li_error_vs_post": float(np.nanmean(e_post)),
        "li_error_vs_pre": float(np.nanmean(e_pre)),
        "edit_index_union": float(np.nanmean(ei_u)),
        "edit_index_symdiff": float(np.nanmean(ei_s)),
        "legal_mass": float(
            np.mean([probs[i, L].sum() for i, L in enumerate(bench.legal_post) if L])
        ),
        "n_scored": int(np.isfinite(e_post).sum()),
        "li_error_vs_post_per_case": e_post.tolist(),
        "edit_index_union_per_case": ei_u.tolist(),
    }
