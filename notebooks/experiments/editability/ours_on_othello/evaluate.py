"""Gates, probes and interventions for OUR transformer trained on Othello.

Nothing here re-derives a metric. The probe grid, the edit specs, the descent, the board
probabilities, the Li error and the Edit Index all come from `othello_transfer/`
(`transfer_pipeline`, `othello_probe`, `othello_data`, `linear_intervention`), which are the
same modules that produced the numbers on Li et al.'s own checkpoint. The only work this
file does is (1) tell those modules that our model has 5 residual points rather than 9,
(2) keep our probe cache separate from theirs, and (3) add the **held-out generalisation
gates**, which `othello_transfer` never needed because it inherited a trained checkpoint.

Why the gates exist and what they are for
-----------------------------------------
Arm `M` sees 90,000 games 300 times over. Training loss will go somewhere memorisation can
reach, so *training* loss says nothing. The question Sevan posed is whether what it learns
**generalises**: `runs/transformers/W16` also overfits its training loss on discworld (best
val at epoch ~40 of 300) and is nevertheless a good predictor on the held-out test split.
So every gate here is computed at **`best_model.pt` on games from a disjoint index range**
(`corpus.TEST_LO`), never on training games:

| gate | formula | units | better | reference |
|---|---|---|---|---|
| **legal-move mass** | mean over positions of `sum_{m in legal(t)} p(m | history)` | 0…1 | ↑ | Li et al. report **0.9998** for their model |
| **top-1 legal rate** | fraction of positions where `argmax p` is a legal move | 0…1 | ↑ | 1.0 for a model that has learned the rules |
| **top-1 accuracy** | fraction where `argmax p` is the move actually played | 0…1 | ↑ | **bounded well below 1**: the generator picks uniformly at random from the legal set, so the Bayes-optimal predictor scores `mean(1/|legal|)`, reported alongside as `bayes_top1` |
| **held-out CE** | cross-entropy in nats over non-pad positions | nats | ↓ | the uniform-over-legal predictor's CE, `mean(log|legal|)`, reported as `bayes_ce` |

The last two matter: on this data a *perfect* model cannot exceed the Bayes rate, so raw
accuracy is uninterpretable without it. `bayes_top1` and `bayes_ce` are computed from the
same held-out games and are the honest ceiling.
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
_XFER = _HERE.parent / "othello_transfer"
for _p in (str(_HERE), str(_XFER), str(_HERE.parent / "othello_gpt"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_data as od  # noqa: E402
import othello_probe as op  # noqa: E402
import transfer_pipeline as tp  # noqa: E402

import corpus as cp  # noqa: E402  (ours)
from model import BLOCK, build  # noqa: E402  (ours)

RUNS = _REPO / "runs" / "ours_on_othello"
DEV = tp.DEVICE
IGNORE = -100


# ── plumbing ──────────────────────────────────────────────────────────────────


def fingerprint(model) -> str:
    """12 hex chars derived from every parameter — a model's identity, cheaply."""
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def attach(model) -> int:
    """Point the shared pipeline at OUR model's geometry and OUR per-model probe cache.

    `transfer_pipeline.N_POINTS` is a module constant (9, for their 8-block model) read by
    `fit_probe_grid`, `run_arm` and — bound at import time — `linear_intervention`. Our
    model has `n_layers + 1` points, so both have to be told.

    ⛔ **The cache directory is per-model, keyed on a hash of the weights.**
    `fit_probe_grid`'s own cache key covers the probe settings and the data but **not the
    model** — it never had to, because `othello_transfer` only ever probed one checkpoint. Here
    we probe several, and on 2026-08-21 a smoke run served the random-init control the *trained*
    model's probes: both reported error 37.08%–57.94%, identically, which would have silently
    destroyed the only baseline that makes an absolute probe error interpretable. The
    fingerprint makes a stale hit impossible rather than unlikely.
    """
    n_points = model.cfg.n_layers + 1
    tp.N_POINTS = n_points
    tp.CACHE_DIR = RUNS / "probe_cache" / fingerprint(model)
    tp.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if "linear_intervention" in sys.modules:
        sys.modules["linear_intervention"].N_POINTS = n_points
    return n_points


def load_run(name: str, which: str = "best_model.pt"):
    """Rebuild the model from its checkpoint and put the pipeline in step with it."""
    ck = torch.load(RUNS / name / which, map_location=DEV, weights_only=False)
    c = ck["model_config"]
    m = build(c["d_model"], c["n_layers"], c["n_heads"], c["window"], c["mlp_ratio"]).to(DEV)
    m.load_state_dict(ck["model_state"])
    m.eval()
    attach(m)
    return m, ck


def random_init(like, seed: int = 0) -> object:
    """Li et al.'s own `--random` control: identical architecture, untrained weights.

    **Seeded.** Without this the control draws new weights on every execution, so its weight
    fingerprint — and therefore its probe cache — changes every run, and the 20-minute probe
    grid is refit each time for a *different* baseline. A control that moves between runs is
    not a control.
    """
    c = like.cfg
    g = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        m = build(c.d_model, c.n_layers, c.n_heads, c.window, c.mlp_ratio).to(DEV)
    finally:
        torch.random.set_rng_state(g)
    m.eval()
    return m


# ── held-out generalisation gates ─────────────────────────────────────────────


def legal_sets(tokens: np.ndarray, lengths: np.ndarray) -> list[list[list[int]]]:
    """Per game, per position, the legal moves as BOARD SQUARES, replayed with their rules."""
    if str(od.OTHELLO_ROOT) not in sys.path:
        sys.path.insert(0, str(od.OTHELLO_ROOT))
    from data.othello import OthelloBoardState

    itos = {v: k for k, v in od.canonical_vocab().items()}
    out = []
    for row, L in zip(tokens, lengths):
        b = OthelloBoardState()
        per = []
        for t in range(int(L)):
            b.umpire(itos[int(row[t])])
            per.append(sorted(b.get_valid_moves()))
        out.append(per)
    return out


@torch.no_grad()
def gates(model, tokens: np.ndarray, lengths: np.ndarray, batch: int = 512,
          log=print) -> dict:
    """Every held-out number, plus the Bayes ceilings the data itself imposes."""
    stoi = od.canonical_vocab()
    legal = legal_sets(tokens, lengths)
    mass, hit1, acc1, ce, bce, btop1, n = 0.0, 0, 0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(tokens), batch):
        tk = torch.from_numpy(tokens[i : i + batch]).long().to(DEV)
        lg = model.logits(tk[:, :BLOCK])
        p = torch.softmax(lg[:, :, 1:], -1).cpu().numpy()  # drop the pad logit
        am = p.argmax(-1) + 1                              # back into token space
        for r in range(len(tk)):
            L = int(lengths[i + r])
            for t in range(L - 1):                         # position t predicts move t+1
                lm = legal[i + r][t]
                if not lm:
                    continue
                toks = [stoi[s] for s in lm]
                pm = float(p[r, t, [k - 1 for k in toks]].sum())
                mass += pm
                hit1 += int(am[r, t] in toks)
                acc1 += int(am[r, t] == int(tokens[i + r, t + 1]))
                ce += -float(np.log(max(p[r, t, int(tokens[i + r, t + 1]) - 1], 1e-12)))
                bce += float(np.log(len(lm)))
                btop1 += 1.0 / len(lm)
                n += 1
        if log and i % (batch * 4) == 0:
            log(f"    gates {i + len(tk):,}/{len(tokens):,}")
    return {"legal_mass": mass / n, "top1_legal": hit1 / n, "top1_acc": acc1 / n,
            "ce": ce / n, "bayes_ce": bce / n, "bayes_top1": btop1 / n,
            "n_positions": n, "n_games": len(tokens)}


def gate_table(rows: dict[str, dict]) -> str:
    """Markdown, with Li et al.'s published legal-move mass on the same axis."""
    h = ("| model | legal-move mass ↑ | top-1 legal ↑ | top-1 acc ↑ | *Bayes top-1* | "
         "CE ↓ | *Bayes CE* | CE excess |")
    out = [h, "|---|---|---|---|---|---|---|---|"]
    for k, g in rows.items():
        out.append(f"| {k} | **{g['legal_mass']:.4f}** | {g['top1_legal']:.4f} | "
                   f"{g['top1_acc']:.4f} | *{g['bayes_top1']:.4f}* | {g['ce']:.4f} | "
                   f"*{g['bayes_ce']:.4f}* | {g['ce'] - g['bayes_ce']:+.4f} |")
    out.append("| *Li et al. (published, their 25.3M-param model)* | *0.9998* | — | — | — | "
               "— | — | — |")
    return "\n".join(out)


# ── probes ────────────────────────────────────────────────────────────────────


def probe_data(n_games: int = cp.PROBE_N) -> od.ProbeData:
    """Board-state labels on the held-out probe corpus — a disjoint index range.

    ⛔ `only=("probe",)` is load-bearing. Without it `cp.build` walks its whole plan and, if the
    20M training pool does not yet exist, starts generating it — 3 h of CPU, from a call that
    looks like a cheap lookup. Cost me a smoke-test run on 2026-08-21.
    """
    tok, ln = cp.load(cp.build(cp.LADDER["D"], log=lambda s: None, only=("probe",))["probe"])
    itos = {v: k for k, v in od.canonical_vocab().items()}
    games = [[itos[int(t)] for t in row[:L]] for row, L in zip(tok[:n_games], ln[:n_games])]
    return od.tokens_and_labels(games)


def linear_probes(grid, target: str = "mine", split: str = "frame") -> dict:
    """{residual point -> the linear probe}, for the direction-addition editors."""
    return {p: grid.probes[(target, "linear", split, p)]
            for p in range(tp.N_POINTS)}


def mlp_probes(grid, target: str = "state", split: str = "frame",
               family: str = "MLP 512 hidden") -> dict:
    return {p: grid.probes[(target, family, split, p)] for p in range(tp.N_POINTS)}


# ── interventions ─────────────────────────────────────────────────────────────


def case_targets(bench):
    """Per case, the intervened tile's CURRENT and TARGET label in mine/theirs coordinates.

    Imported behaviour, restated here only so this module does not depend on
    `linear_intervention` having been imported first. The benchmark flips absolute colour;
    the player to move does not change, so flipping colour flips MINE<->THEIRS exactly.
    """
    if str(od.OTHELLO_ROOT) not in sys.path:
        sys.path.insert(0, str(od.OTHELLO_ROOT))
    from data.othello import OthelloBoardState

    with open(od.OTHELLO_ROOT / "intervention_benchmark.pkl", "rb") as f:
        ds = pickle.load(f)
    cur = np.zeros(bench.n_cases, np.int64)
    for i, c in enumerate(ds):
        b = OthelloBoardState()
        b.update(c["history"], prt=False)
        nxt = 2 if b.next_hand_color > 0 else 0
        cur[i] = 1 if c["ori_color"] == nxt else 2  # MINE / THEIRS
    return cur, np.where(cur == 1, 2, 1)


def unsteered(model, bench):
    return od.scorecard(tp.unsteered(model, bench), bench)


__all__ = ["attach", "fingerprint", "load_run", "random_init", "gates", "gate_table", "legal_sets",
           "probe_data", "linear_probes", "mlp_probes", "case_targets", "unsteered",
           "op", "od", "tp", "cp", "RUNS", "DEV"]
