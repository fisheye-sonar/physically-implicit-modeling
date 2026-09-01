"""Othello gates, probe grid, and editor arms — the counterpart of discworld's bench.py.

Ported 2026-08-31 from ``ours_on_othello/evaluate.py`` (gates), ``othello_transfer/
transfer_pipeline.py`` (probe grid, GS arm) and ``othello_transfer/linear_intervention.py``
(ND + PI arms), rebuilt on the canonical parts: probes from ``pim.probes``, editors from
``pim.editors``, scoring from ``pim.metrics.othello_moves``. One deliberate upgrade over
the originals: the probe-grid cache now keys on the MODEL FINGERPRINT (``pim.probes.cache``),
closing the same 2026-08-21 hole here that ``othello_arch`` had already closed on its side.

Every measurement is step-0 (no rollout): Othello rollout semantics were never designed,
and the models raise on ``predict_step`` to keep it that way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook
from pim.editors.pinv import inject_state
from pim.environments.othello.bench import Benchmark
from pim.environments.othello.data import N_CLASSES, N_TILES, board_probs, canonical_vocab
from pim.environments.othello.vendor.othello import OthelloBoardState
from pim.metrics.othello_moves import move_scorecard
from pim.probes.base import fit_probe
from pim.probes.cache import ProbeCache
from pim.probes.mlp import CANONICAL_HIDDEN

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK = 59

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
PROBE_CACHE = ProbeCache(_REPO / "runs" / "probe_cache" / "othello")


# ── held-out generalisation gates ────────────────────────────────────────────


def legal_sets(tokens: np.ndarray, lengths: np.ndarray) -> list[list[list[int]]]:
    """Per game, per position, the legal moves as BOARD SQUARES, replayed with their rules."""
    itos = {v: k for k, v in canonical_vocab().items()}
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
    """Every held-out number, plus the Bayes ceilings the data itself imposes.

    The generator draws uniformly from the legal set, so ``bayes_ce = E[log|legal|]``
    and ``bayes_top1 = E[1/|legal|]`` are exact; the meaningful training quantity is
    the CE EXCESS over bayes_ce, never raw accuracy.
    """
    stoi = canonical_vocab()
    legal = legal_sets(tokens, lengths)
    mass, hit1, acc1, ce, bce, btop1, n = 0.0, 0, 0, 0.0, 0.0, 0.0, 0
    for i in range(0, len(tokens), batch):
        tk = torch.from_numpy(tokens[i: i + batch]).long().to(DEV)
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
                mass += float(p[r, t, [k - 1 for k in toks]].sum())
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


# ── probes over residual points ──────────────────────────────────────────────


def _split(n_seq: int, seq_of_row: np.ndarray, how: str, holdout: float, seed: int):
    """``"frame"`` = Li's pooled-row split (their paper's convention, kept for the
    replication anchor); ``"sequence"`` = whole games held out, this repo's rule.
    Numbers from the two are NOT comparable and are always labelled."""
    rng = np.random.default_rng(seed)
    n_rows = len(seq_of_row)
    if how == "frame":
        perm = rng.permutation(n_rows)
        cut = int(round((1 - holdout) * n_rows))
        return perm[:cut], perm[cut:]
    order = rng.permutation(n_seq)
    is_tr = np.zeros(n_seq, bool)
    is_tr[order[: int(round((1 - holdout) * n_seq))]] = True
    tr_mask = is_tr[seq_of_row]
    return np.where(tr_mask)[0], np.where(~tr_mask)[0]


@dataclass
class ProbeGrid:
    probes: dict  # (target, family, split, point) -> WorldStateProbe
    stats: list


def fit_probe_grid(model, data, *, targets=("mine",),
                   families=("linear", "mlp"), splits=("sequence",),
                   holdout: float = 0.2, epochs: int = 200, batch: int = 4096,
                   lr: float = 1e-3, seed: int = 0, log=print,
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

    # per-run home when the caller names one (canonical scoring passes the run's own
    # probes/ dir); the shared pool is only the ad-hoc fallback — see bench.fit_probes.
    store = ProbeCache(cache_dir) if cache_dir is not None else PROBE_CACHE
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

    seq_of_row = np.repeat(np.arange(len(data.tokens))[:, None],
                           data.tokens.shape[1], 1)[data.mask]
    ys = {"state": data.labels[data.mask].astype(np.int64),
          "mine": data.mine[data.mask].astype(np.int64)}
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
                                          n_classes=N_CLASSES)
                    st |= {"target": target, "family": fam, "split": split,
                           "point": point}
                    probes[(target, fam, split, point)] = probe
                    stats.append(st)
                    if log:
                        log(f"  point {point}  {target:5s}  {split:8s}  {fam:6s}  "
                            f"error {st['error_rate']:6.2f}%  "
                            f"(in-sample {st['error_rate_insample']:5.2f}%)")
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
        probs[ids] = board_probs(model.decode(idx))
    return probs


def unsteered(model, bench: Benchmark) -> dict:
    """No intervention — the −0.829 floor every arm is read against."""
    return move_scorecard(unsteered_probs(model, bench), bench.legal_pre, bench.legal_post)


@torch.no_grad()
def linear_arm(model, bench: Benchmark, probes: dict, tgt_lab, cur_lab, *,
               mode: str, alpha: float, points) -> tuple[np.ndarray, dict]:
    """ND and PI on the classification probes, exactly as ``linear_intervention.run``.

    mode "add"      ND: the probe weight row for (tile, target class), standardised
                    (w / x_std — the raw-space gradient), unit-normed, scaled by α·‖x‖.
    mode "add_sub"  ND target−current: subtract the current class's row first.
    mode "pinv"     PI: our canonical editor. The probe is linear in its STANDARDISED
                    input, so the injection is solved in z-space and mapped back —
                    exactly ``pim.editors.pinv``'s "zspace" (classification probes have
                    identity y-affine, so there is no affine question here). The target
                    logits swap the intervened tile's current↔target class scores.
    """
    probs = np.zeros((bench.n_cases, N_TILES), np.float32)
    ratios = []
    pinv_ops = {}
    if mode == "pinv":
        for ell, p in probes.items():
            A = p.net.weight.detach()
            pinv_ops[ell] = (A, torch.linalg.pinv(A), p.net.bias.detach())
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
            if mode in ("add", "add_sub"):
                W = p.net.weight.detach().view(N_TILES, N_CLASSES, -1)
                d = W[sq, td] / p.x_std
                if mode == "add_sub":
                    d = d - (W[sq, cd] / p.x_std)
                d = d / d.norm(dim=1, keepdim=True)
                # α is a FRACTION OF THE ACTIVATION NORM, so one value means the same
                # size of write at every residual point (the scale differs ~3×)
                delta = alpha * cur.norm(dim=1, keepdim=True) * d
            else:
                A, Ap, bv = pinv_ops[layer]
                z = (cur - p.x_mean) / p.x_std
                lg = p.net(z).view(bsz, N_TILES, N_CLASSES).clone()
                sel = lg[torch.arange(bsz), sq]
                new = sel.clone()
                new[torch.arange(bsz), td] = sel[torch.arange(bsz), cd]
                new[torch.arange(bsz), cd] = sel[torch.arange(bsz), td]
                lg[torch.arange(bsz), sq] = new
                z_new = inject_state(z, lg.view(bsz, -1), A, Ap, bv)
                delta = alpha * (z_new - z) * p.x_std
            _rec.append(float((delta.norm(dim=1) / cur.norm(dim=1)).mean()))
            out = x.clone()
            out[:, -1] = cur + delta
            return out

        probs[ids] = board_probs(model.decode(idx, edit=hook))
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
        tv = torch.zeros(bsz, N_TILES, dtype=torch.long, device=DEV)
        tv[torch.arange(bsz), torch.from_numpy(bench.pos_int[ids]).to(DEV)] = (
            torch.from_numpy(lab[ids]).to(DEV))
        specs = {ell: build_edit_spec(probes[ell], x0[ell], cm, tv, beta=beta)
                 for ell in range(n_points)}
        hook = make_intervention_hook(probes, specs, start_layer, alpha=alpha,
                                      n_steps=n_steps, optimizer=optimizer)
        with torch.no_grad():
            probs[ids] = board_probs(model.decode(idx, edit=hook))
        del rs, x0, specs
    return probs, move_scorecard(probs, bench.legal_pre, bench.legal_post)
