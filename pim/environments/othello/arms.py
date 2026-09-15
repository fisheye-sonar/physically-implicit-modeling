"""Othello gates, probe grid, and editor arms — the counterpart of discworld's bench.py.

Ported 2026-08-31 from ``ours_on_othello/evaluate.py`` (gates), ``othello_transfer/
transfer_pipeline.py`` (probe grid, GS arm) and ``othello_transfer/linear_intervention.py``
(ND + PI arms), rebuilt on the canonical parts: probes from ``pim.probes``, editors from
``pim.editors``, scoring from ``pim.metrics.set_editability``. One deliberate upgrade over
the originals: the probe-grid cache now keys on the MODEL FINGERPRINT (``pim.probes.cache``),
closing the same 2026-08-21 hole here that ``othello_arch`` had already closed on its side.

Every measurement is step-0 (no rollout): Othello rollout semantics were never designed,
and the models raise on ``predict_step`` to keep it that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.nanda import addition_delta, probe_direction
from pim.editors.pinv import pinv_step, swap_class_logits
from pim.environments.othello.bench import Benchmark
from pim.environments.othello.data import (
    MINE, N_CLASSES, N_TILES, REGRESSION_TARGETS, T_MODEL, board_probs, canonical_vocab,
    flatten_rows, move_probs, signed_mine)
from pim.metrics.decodability import probe_skill_from_stats
from pim.environments.othello.vendor.othello import OthelloBoardState
from pim.metrics.set_editability import move_scorecard
from pim.probes.base import CANONICAL_HIDDEN, FIT_BATCH, FIT_EPOCHS, FIT_LR, fit_probe
from pim.probes.cache import ProbeCache

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = T_MODEL

_REPO = Path(__file__).resolve().parents[3]


def _require_cache_dir(cache_dir) -> Path:
    if cache_dir is None:
        raise ValueError("cache_dir is required: every fitted probe is persisted in a named "
                         "directory (the run's probes/ or the experiment's) — no shared pool")
    return Path(cache_dir)


# ── held-out generalisation gates ────────────────────────────────────────────


def legal_sets(tokens: np.ndarray, lengths: np.ndarray, flip: bool = True,
               placement: str = "enclosure") -> list[list[list[int]]]:
    """Per game, per position, the legal moves as BOARD SQUARES, replayed with their rules."""
    itos = {v: k for k, v in canonical_vocab().items()}
    out = []
    for row, L in zip(tokens, lengths):
        b = OthelloBoardState(flip=flip, placement=placement)
        per = []
        for t in range(int(L)):
            b.umpire(itos[int(row[t])])
            per.append(sorted(b.get_valid_moves()))
        out.append(per)
    return out


@torch.no_grad()
def gates(model, tokens: np.ndarray, lengths: np.ndarray, batch: int = 512,
          log=print, flip: bool = True, placement: str = "enclosure") -> dict:
    """Every held-out number, plus the Bayes ceilings the data itself imposes.

    The generator draws uniformly from the legal set, so ``bayes_ce = E[log|legal|]``
    and ``bayes_top1 = E[1/|legal|]`` are exact; the meaningful training quantity is
    the CE EXCESS over bayes_ce, never raw accuracy.
    """
    stoi = canonical_vocab()
    legal = legal_sets(tokens, lengths, flip, placement)
    # The head's output convention (see `data.move_probs`): the canonical CE runs are
    # "logits"; an MSE-on-one-hot head is "raw" — its outputs are used as they are, so
    # `legal_mass` and `ce` below are only distribution-valid for "logits"/"clipnorm";
    # `out_sum_mean` / `out_neg_mass_mean` say how far from a distribution the raw head is.
    kind = getattr(model, "output_kind", "logits")
    mass, hit1, acc1, ce, bce, btop1, n = 0.0, 0, 0, 0.0, 0.0, 0.0, 0
    osum, oneg = 0.0, 0.0
    for i in range(0, len(tokens), batch):
        tk = torch.from_numpy(tokens[i: i + batch]).long().to(DEV)
        lg = model.logits(tk[:, :BLOCK])
        pt = move_probs(lg, kind)
        p = pt.cpu().numpy()                               # (B, T, 60), pad output dropped
        am = p.argmax(-1) + 1                              # back into token space
        for r in range(len(tk)):
            L = int(lengths[i + r])
            for t in range(L - 1):                         # position t predicts move t+1
                lm = legal[i + r][t]
                if not lm:
                    continue
                toks = [stoi[s] for s in lm]
                mass += float(p[r, t, [k - 1 for k in toks]].sum())
                hit1 += int(am[r, t] in toks)
                acc1 += int(am[r, t] == int(tokens[i + r, t + 1]))
                ce += -float(np.log(max(p[r, t, int(tokens[i + r, t + 1]) - 1], 1e-12)))
                bce += float(np.log(len(lm)))
                btop1 += 1.0 / len(lm)
                osum += float(p[r, t].sum())
                oneg += float(-p[r, t][p[r, t] < 0].sum())
                n += 1
        if log and i % (batch * 4) == 0:
            log(f"    gates {i + len(tk):,}/{len(tokens):,}")
    return {"legal_mass": mass / n, "top1_legal": hit1 / n, "top1_acc": acc1 / n,
            "ce": ce / n, "bayes_ce": bce / n, "bayes_top1": btop1 / n,
            "n_positions": n, "n_games": len(tokens), "output_kind": kind,
            "out_sum_mean": osum / n, "out_neg_mass_mean": oneg / n}


# ── probes over residual points ──────────────────────────────────────────────


def _split(n_seq: int, seq_of_row: np.ndarray, how: str, holdout: float, seed: int):
    """``"frame"`` = Li's pooled-row split (their paper's convention, kept for the
    replication anchor); ``"sequence"`` = whole games held out, this repo's rule.
    Numbers from the two are NOT comparable and are always labelled."""
    rng = np.random.default_rng(seed)
    n_rows = len(seq_of_row)
    if how == "frame":
        perm = rng.permutation(n_rows)
        cut = int((1 - holdout) * n_rows)
        return perm[:cut], perm[cut:]
    order = rng.permutation(n_seq)
    is_tr = np.zeros(n_seq, bool)
    is_tr[order[: int((1 - holdout) * n_seq)]] = True       # int(), as bench.fit_probes
    tr_mask = is_tr[seq_of_row]
    return np.where(tr_mask)[0], np.where(~tr_mask)[0]


def observation_probes(data, family: str = "linear", target: str = "mine",
                       holdout: float = 0.2, seed: int = 0, cache_dir=None,
                       cache: bool = True, log=print, epochs: int | None = None,
                       align: str = "left") -> tuple:
    """The OBSERVATION floor: the canonical probes fitted to the causal MOVE history
    instead of a model's residual stream. No model is involved.

    Matched to ``fit_probe_grid`` in every other respect — same games, same target
    frame, same SEEDed sequence split (identical permutation, so the held-out games are
    literally the same ones), same probe families, same padding mask.

    ⚠ **Read this floor differently from discworld's.** An Othello board is a
    DETERMINISTIC function of the move sequence, so the state is perfectly recoverable
    from these features in principle. A low number here therefore means "a linear map /
    one hidden layer cannot compute the flip rules from raw moves", never "the
    information is not in the input". Discworld's observation is genuinely lossy by
    comparison (noise, and depth is never directly observed).

    Returns ``(probe, stats)`` — ONE probe; there is no residual point to sweep.
    """
    import torch as _t

    from pim.environments.othello.data import canonical_vocab
    from pim.probes.baselines import CausalHistory, fit_baseline_probe
    from pim.probes.mlp import CANONICAL_HIDDEN

    store = ProbeCache(_require_cache_dir(cache_dir))
    n_seq = int(len(data.tokens))
    vocab = len(canonical_vocab())
    extra = {} if epochs is None else {"epochs": int(epochs)}   # see discworld.arms
    if align != "left":                     # existing (left-aligned) keys stay as they are
        extra["align"] = align
    fname, prov = store.key(None, kind="othello_observation", target=target,
                            family=family, holdout=holdout, seed=seed, n_seq=n_seq,
                            n_rows=int(data.mask.sum()), vocab=vocab, **extra)
    if cache:
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            if log:
                log(f"    obs-baseline cache HIT  {fname}")
            return hit
    # the same permutation _split draws for "sequence" — identical held-out games
    order = np.random.default_rng(seed).permutation(n_seq)
    cut = int((1 - holdout) * n_seq)
    tr, te = order[:cut], order[cut:]

    regress = target in REGRESSION_TARGETS
    if regress:                                   # the signed mine/theirs values (2026-09-09)
        y_t = _t.from_numpy(signed_mine(data.mine)).to(DEV)
    else:
        y = data.mine if target == "mine" else data.labels
        y_t = _t.from_numpy(y.astype("int64")).to(DEV)
    hist = CausalHistory(_t.from_numpy(data.tokens).to(DEV), kind="one_hot", vocab=vocab, align=align)
    out = fit_baseline_probe(
        hist, y_t, tr, te,
        hidden=None if family == "linear" else CANONICAL_HIDDEN,
        n_classes=None if regress else 3,
        row_mask=_t.from_numpy(data.mask).to(DEV), seed=seed, log=log,
        **{k: v for k, v in extra.items() if k != "align"})
    if log:
        st = out[1]
        log(f"    obs baseline [{target}/{family}]: skill {probe_skill_from_stats(st):+.4f} "
            f"(d_in {st['d_in']})")
    if cache:
        store.store(fname, prov, out)
    return out


@dataclass
class ProbeGrid:
    probes: dict  # (target, family, split, point) -> WorldStateProbe
    stats: list


def fit_probe_grid(model, data, *, targets=("mine",),
                   families=("linear", "mlp"), splits=("sequence",),
                   holdout: float = 0.2, epochs: int = FIT_EPOCHS, batch: int = FIT_BATCH,
                   lr: float = FIT_LR, seed: int = 0, log=print,
                   cache: bool = True, cache_dir=None) -> ProbeGrid:
    """One probe per (target, family, split, residual point). Cached with the model
    fingerprint in the key. ``family`` "mlp" = the canonical MLP-128 (Li's own shape
    for classification). Harvests one residual point at a time (~2.4 GB, not 22).

    Defaults narrowed 2026-09-01 from 72 fits to 18, both cuts settled by measurement:

    * ``targets=("mine",)`` — mine/theirs only. Absolute colour ("state") is Li et al.'s
      original frame; Nanda showed it is not linearly decodable while mine/theirs is, and
      once the GS target-frame bug was fixed every editor's best arm read mine/theirs
      probes. Nothing needs ``state`` any more.
    * ``splits=("sequence",)`` — whole games held out, this repo's anti-leak rule. The
      ``frame`` split (Li's pooled-row convention, kept for a while as the anchor to
      their published tables) measured 0.976 against sequence's 0.975 at 20k games, so
      the leak does not bite at this corpus size and the honest split costs nothing.

    Both axes remain arguments: pass them explicitly to reproduce an older grid or to
    re-measure either claim.
    """
    from pim.environments.othello.data import harvest_point

    store = ProbeCache(_require_cache_dir(cache_dir))     # the run's or the experiment's probes/
    n_points = model.n_layers + 1
    fname, prov = store.key(
        model, kind="othello_grid", targets=list(targets), families=list(families),
        splits=list(splits), holdout=holdout, epochs=epochs, batch=batch, lr=lr,
        seed=seed, n_seq=int(len(data.tokens)), n_rows=int(data.mask.sum()),
        n_points=n_points)
    if cache:
        blob = store.load(fname, prov, device=DEV)
        if blob is not None:
            if log:
                log(f"  probe grid cache HIT ({fname})")
            return ProbeGrid(blob["probes"], blob["stats"])

    seq_of_row, _ = flatten_rows(data)
    # 3-way labels for the categorical targets; the signed ±1/0 values for "mine_signed",
    # which is fitted by REGRESSION (n_classes None) — the Othello counterpart of the
    # discworld grid target's question, asked the other way round (2026-09-09)
    ys = {t: (flatten_rows(data, t)[1] if t in REGRESSION_TARGETS
              else flatten_rows(data, t)[1].astype(np.int64)) for t in targets}
    idx = {s: _split(len(data.tokens), seq_of_row, s, holdout, seed) for s in splits}
    hidden = {"linear": None, "mlp": CANONICAL_HIDDEN}

    probes, stats = {}, []
    for point in range(n_points):
        acts = harvest_point(model, data.tokens, point)
        x = acts[data.mask]
        del acts
        for target in targets:
            y = ys[target]
            for split in splits:
                tr, te = idx[split]
                for fam in families:
                    probe, st = fit_probe(x[tr], y[tr], x[te], y[te],
                                          hidden=hidden[fam], epochs=epochs,
                                          batch=batch, lr=lr, device=DEV, seed=seed,
                                          n_classes=None if target in REGRESSION_TARGETS
                                          else N_CLASSES)
                    st |= {"target": target, "family": fam, "split": split,
                           "point": point}
                    probes[(target, fam, split, point)] = probe
                    stats.append(st)
                    if log:
                        log(f"  point {point}  {target:11s}  {split:8s}  {fam:6s}  "
                            f"skill {probe_skill_from_stats(st):+.4f}")
        del x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if cache:
        store.store(fname, prov, {"probes": probes, "stats": stats})
    return ProbeGrid(probes, stats)


# ── the editor arms (step-0, over the 1001-case bench) ───────────────────────


@torch.no_grad()
def unsteered_probs(model, bench: Benchmark) -> np.ndarray:
    """(1001, 64) no-intervention move distributions.

    Both the −0.829 Edit Index floor and the guard's DENOMINATOR
    (``move_fidelity_ratio``) are computed from these, so the baseline exists once.
    """
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        probs[ids] = board_probs(model.decode(idx), getattr(model, "output_kind", "logits"))
    return probs


def unsteered(model, bench: Benchmark) -> dict:
    """No intervention — the −0.829 floor every arm is read against."""
    return move_scorecard(unsteered_probs(model, bench), bench.legal_pre, bench.legal_post)


@torch.no_grad()
def linear_arm(model, bench: Benchmark, probes: dict, tgt_lab, cur_lab, *,
               mode: str, alpha: float, points) -> tuple[np.ndarray, dict]:
    """ND and PI on the classification probes — ``pim.editors.nanda`` / ``pim.editors.pinv``
    called with the Othello case structure; NOTHING is re-derived here (the inline copies
    this function carried until 2026-09-07 are pinned equal in
    ``tests/test_editors_canonical.py``).

    mode "add"      ND: the probe weight row for (tile, target class), standardised
                    (w / x_std — the raw-space gradient), unit-normed, scaled by α·‖x‖ —
                    ``probe_direction(per_sample=True)`` + ``addition_delta``.
    mode "add_sub"  ND target−current: subtract the current class's row first.
    mode "pinv"     PI: ``pinv_step`` in z-space (a classification probe has no y-affine,
                    so there is no affine question here). The target is the probe's own
                    read-out with the intervened tile's current↔target class scores swapped.
    """
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    ratios = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        bsz = len(ids)
        sq = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        td = torch.from_numpy(tgt_lab[ids]).to(DEV)
        cd = torch.from_numpy(cur_lab[ids]).to(DEV)
        rec = []

        def hook(layer, x, _rec=rec):
            if layer not in points:
                return x
            p = probes[layer]
            cur = x[:, -1]
            if p.n_classes is None:
                # the signed mine/theirs REGRESSION probe (2026-09-09): the flip asks the
                # tile's value to become +1 (mine) or −1 (theirs). ND: the tile's probe row,
                # signed by the requested direction — the target−current contrast, whose
                # magnitude is the same 2 units for every case. PI: the probe's own 64-value
                # read-out with the tile set to its target, solved in z-space with the
                # y-affine (the discworld regression path, one tile driven per case).
                ar = torch.arange(bsz, device=cur.device)
                val = torch.where(td == MINE, 1.0, -1.0).to(cur.dtype)
                if mode in ("add", "add_sub"):
                    d = probe_direction(p, sq, per_sample=True) * val[:, None]
                    delta = addition_delta(cur, d, alpha)
                else:
                    tgt = p(cur).clone()
                    tgt[ar, sq] = val
                    delta = alpha * pinv_step(cur, tgt, p, space="zspace")
            elif mode in ("add", "add_sub"):
                # flat probe row of (tile, class) = tile * N_CLASSES + class
                d = probe_direction(p, sq * N_CLASSES + td, per_sample=True,
                                    subtract_rows=(sq * N_CLASSES + cd) if mode == "add_sub" else None)
                # α is a FRACTION OF THE ACTIVATION NORM, so one value means the same
                # size of write at every residual point (the scale differs ~3×)
                delta = addition_delta(cur, d, alpha)
            else:
                # the probe's own read-out with current <-> target swapped at the square
                # (the shared spelling of a categorical flip — discworld's grid target
                # calls the same helper twice, once per cell)
                lg = swap_class_logits(p(cur), sq, cd, td)   # (B, N_TILES, N_CLASSES)
                delta = alpha * pinv_step(cur, lg.view(bsz, -1), p, space="zspace")
            _rec.append(float((delta.norm(dim=1) / cur.norm(dim=1)).mean()))
            out = x.clone()
            out[:, -1] = cur + delta
            return out

        probs[ids] = board_probs(model.decode(idx, edit=hook),
                                 getattr(model, "output_kind", "logits"))
        ratios.append(np.mean(rec) if rec else 0.0)
    card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
    card["write_ratio"] = float(np.mean(ratios))
    return probs, card


def grad_steer_arm(model, bench: Benchmark, probes: dict, start_layer: int, *,
                   alpha: float, n_steps: int, beta: float,
                   optimizer: str = "adam",
                   target_labels=None) -> tuple[np.ndarray, dict]:
    """GS over the 1001 cases — ``transfer_pipeline.run_arm``, on the canonical parts.

    Bucket by bucket, because the intervention hook writes ``x[:, -1]`` and every row
    in a batch must have its last real move at the same index.

    ``target_labels`` selects the TARGET FRAME the descent aims for, and must match
    the frame of the probes steered through. Default None = ``bench.new_class``
    (absolute colour, Li §4.1 verbatim — pair with the ``state`` probes). Pass the
    mine-coordinate targets from ``case_targets(bench)[1]`` to steer through the
    ``mine`` probes instead — the open question (2026-08-31) is whether GS is dead
    in that frame or the old negative was an artefact of the pre-canonical probes.
    """
    n_points = model.n_layers + 1
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEV)
        bsz = len(ids)
        with torch.no_grad():
            rs = model.residual_stack(idx)
        x0 = {ell: rs[ell][:, -1] for ell in range(n_points)}
        cm = np.zeros((bsz, N_TILES), bool)
        cm[np.arange(bsz), bench.pos_int[ids]] = True
        lab = bench.new_class if target_labels is None else np.asarray(target_labels)
        sq_t = torch.from_numpy(bench.pos_int[ids]).to(DEV)
        if next(iter(probes.values())).n_classes is None:
            # regression probes (signed mine/theirs): the tile's target VALUE, ±1
            tv = torch.zeros(bsz, N_TILES, device=DEV)
            tv[torch.arange(bsz), sq_t] = torch.where(
                torch.from_numpy(lab[ids]).to(DEV) == MINE, 1.0, -1.0)
        else:
            tv = torch.zeros(bsz, N_TILES, dtype=torch.long, device=DEV)
            tv[torch.arange(bsz), sq_t] = torch.from_numpy(lab[ids]).to(DEV)
        specs = {ell: build_edit_spec(probes[ell], x0[ell], cm, tv, beta=beta)
                 for ell in range(n_points)}
        hook = make_intervention_hook(probes, specs, start_layer, alpha=alpha,
                                      n_steps=n_steps, optimizer=optimizer)
        with torch.no_grad():
            probs[ids] = board_probs(model.decode(idx, edit=hook),
                                 getattr(model, "output_kind", "logits"))
        del rs, x0, specs
    return probs, move_scorecard(probs, bench.legal_pre, bench.legal_post)


# ── IM: the inverse-map editor on Othello (2026-09-15) ──────────────────────────────────────


@torch.no_grad()
def inverse_arms(model, bench: Benchmark, data, *, rules: dict, cache_dir, n_games: int,
                 seed: int = 0, k: int | None = None, points=None,
                 uns_probs: np.ndarray | None = None, log=print) -> tuple[list[dict], dict]:
    """IM (h′ = g(board_post) at the last position) and IM-NN (the mean residual of the k
    training boards nearest the target board, Hamming) at every residual point, on the
    canonical cases. g: one-hot mine/theirs board (64 × 3) → residual, the mirror of the
    MLP-128 probe, fitted on ``data``'s rows (the probe games; the same seeded 80/20 split
    BY GAME as ``fit_probe_grid``) and cached in ``cache_dir`` (kind ``inverse_map``); ``rules`` =
    ``corpus.rules_of(instance)``, to replay the bench histories into boards.
    Returns the arm records (canonical scorecard + guard when ``uns_probs`` is given) and
    ``{"g_r2": [...], "g_rmse": [...]}``."""
    from pim.editors.inverse import inverse_overwrite, retrieval_overwrite
    from pim.environments.othello.bench import case_targets
    from pim.environments.othello.data import (N_CLASSES, N_TILES, board_probs, canonical_vocab,
                                               flatten_rows, harvest_point, tokens_and_labels)
    from pim.metrics.set_editability import move_fidelity_ratio, move_scorecard
    from pim.probes.cache import ProbeCache
    from pim.probes.inverse import (INVERSE_EPOCHS, INVERSE_HIDDEN, RETRIEVAL_K, RetrievalBank,
                                    fit_inverse_map)

    store = ProbeCache(_require_cache_dir(cache_dir))
    seq_of_row, states = flatten_rows(data, "mine")                          # (rows,), (rows, 64) ∈ {0,1,2}
    n_seq = int(data.mask.shape[0])
    tr_idx, te_idx = _split(n_seq, seq_of_row, "sequence", 0.2, int(seed))
    onehot = lambda st: np.eye(N_CLASSES, dtype=np.float32)[st].reshape(len(st), -1)   # noqa: E731
    X_all = onehot(states)
    # the bench's pre- and post-edit boards in the mover's frame (the case's tile flipped)
    itos = {v: kk for kk, v in canonical_vocab().items()}
    n_cases = bench.n_cases
    hist = [None] * n_cases
    for toks, ids in zip(bench.tokens, bench.case_ids):
        for row, i in zip(toks, ids):
            hist[i] = [itos[int(t)] for t in row]
    bd = tokens_and_labels([hist[i] for i in range(n_cases)], **rules)   # the instance's rules (corpus.rules_of)
    cur_lab, tgt_lab = case_targets(bench)
    s_pre = np.stack([bd.mine[i, len(hist[i]) - 1] for i in range(n_cases)])
    s_post = s_pre.copy()
    s_post[np.arange(n_cases), bench.pos_int] = tgt_lab
    assert (s_pre[np.arange(n_cases), bench.pos_int] == cur_lab).all(), "pre-edit board disagrees with the bench"
    Xpost_t = torch.from_numpy(onehot(s_post)).to(DEV)
    X_tr_t = torch.from_numpy(X_all[tr_idx]).to(DEV)
    okind = getattr(model, "output_kind", "logits")
    recs, stats = [], {"g_r2": [], "g_rmse": []}
    for ell in (points if points is not None else range(model.n_layers + 1)):
        fname, prov = store.key(model, kind="inverse_map", target="mine-onehot", n_seq=n_seq,
                                split="sequence", seed=int(seed), hidden=INVERSE_HIDDEN,
                                epochs=INVERSE_EPOCHS, point=int(ell), n_games=int(n_games))
        acts = harvest_point(model, data.tokens, ell)
        H = acts[data.mask]
        del acts
        hit = store.load(fname, prov, device=DEV)
        if hit is not None:
            g, st = hit["g"].to(DEV), hit["stats"]
        else:
            g, st = fit_inverse_map(X_all[tr_idx], H[tr_idx], X_all[te_idx], H[te_idx], seed=int(seed), device=DEV)
            store.store(fname, prov, {"g": g, "stats": st})
            if log:
                log(f"    inverse map point {ell}: held-out R² {st['r2']:+.3f}  WROTE {fname}")
        bank = RetrievalBank(X_tr_t, torch.from_numpy(H[tr_idx]).to(DEV), metric="onehot",
                             k=RETRIEVAL_K if k is None else int(k))
        del H
        stats["g_r2"].append(float(st["r2"])); stats["g_rmse"].append(float(st["rmse"]))
        for editor, h_new_all in (("IM", inverse_overwrite(g, Xpost_t)),
                                  ("IM-NN", retrieval_overwrite(bank, Xpost_t))):
            probs = np.zeros((n_cases, N_TILES), np.float32)
            ratios = []
            for toks, ids in zip(bench.tokens, bench.case_ids):
                idx = torch.from_numpy(toks).to(DEV)
                h_new = h_new_all[torch.as_tensor(ids, device=DEV)]

                def hook(layer, x, _h=h_new):
                    if layer != ell:
                        return x
                    cur = x[:, -1]
                    ratios.append(float(((_h - cur).norm(dim=1) / cur.norm(dim=1)).mean()))
                    out = x.clone()
                    out[:, -1] = _h
                    return out
                probs[ids] = board_probs(model.decode(idx, edit=hook), okind)           # THE write
            card = move_scorecard(probs, bench.legal_pre, bench.legal_post)
            rec = {"editor": editor, "point": int(ell), "alpha": 1.0, "g_r2": float(st["r2"]),
                   "write_ratio": float(np.mean(ratios)) if ratios else None,
                   **{kk: v for kk, v in card.items() if isinstance(v, (int, float))}}
            if uns_probs is not None:
                rec["fidelity_ratio"] = move_fidelity_ratio(probs, uns_probs, bench.legal_post)
            if editor == "IM-NN":
                rec["k"] = int(bank.k)
            recs.append(rec)
        del bank
        torch.cuda.empty_cache()
    return recs, stats
