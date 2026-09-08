#!/usr/bin/env python
"""How much of the token model's cross-entropy is irreducible? A frame n-gram floor.

The 8-ray observation is COARSE: a frame history bounds the objects' continuous positions
and velocities but does not pin them, so the next frame is genuinely uncertain given the
token history — the Bayes floor of next-frame CE is NOT zero on this instance, for the
regression model and the token model alike (they see the same 421 patterns). This
estimates an upper bound on that floor from the corpus itself: order-k frame n-gram
conditionals P(next | previous k frames), k = 1..K, counted on the 20M-sequence train
split (~800M transitions) and scored on the 10k test sequences with backoff to the longest
context seen (add-α smoothing, α = 0.01). A model at CE ≈ the best n-gram is at or near
the floor; a model well above it has room; a model BELOW it has learned longer-range
structure than K frames.

Output: experiments/dw_tokens/scores/ngram_floor_<instance>.json (+ a printed table).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld.tokens import load_tokens  # noqa: E402

EXP = REPO / "experiments" / "dw_tokens"
ALPHA = 0.01


def ngram_counts(tok, k: int, V: int, shard: int = 2_000_000, log=print):
    """Sorted unique keys ctx*V + next (ctx = base-V code of the k previous frames) and counts."""
    N, T = tok.shape
    keys, cnts = [], []
    for s in range(0, N, shard):
        x = np.asarray(tok[s: s + shard]).astype(np.int64)
        ctx = np.zeros((len(x), T - k), np.int64)
        for j in range(k):
            ctx = ctx * V + x[:, j: T - k + j]
        u, c = np.unique((ctx * V + x[:, k:]).ravel(), return_counts=True)
        keys.append(u)
        cnts.append(c)
        log(f"    k={k} shard {s // shard + 1}/{-(-N // shard)}: {len(u):,} distinct (ctx, next)", flush=True)
    keys, cnts = np.concatenate(keys), np.concatenate(cnts)
    u, inv = np.unique(keys, return_inverse=True)
    c = np.bincount(inv, weights=cnts).astype(np.int64)
    return u, c


class Order:
    """One n-gram order: P(next | ctx) lookups and the argmax next per context."""

    def __init__(self, k, u, c, V):
        self.k, self.V, self.u, self.c = k, V, u, c
        ctx = u // V
        self.ctx_u, first = np.unique(ctx, return_index=True)
        self.ctx_tot = np.add.reduceat(c, first)
        # argmax next per context: the position of the max count inside each group
        best = np.zeros(len(self.ctx_u), np.int64)
        for g, (lo, hi) in enumerate(zip(first, np.append(first[1:], len(c)))):
            best[g] = u[lo + int(np.argmax(c[lo:hi]))] % V
        self.best = best

    def lookup(self, ctx, nxt):
        """(p_smoothed, seen_ctx, top1) for arrays of ctx codes and next tokens."""
        i = np.searchsorted(self.ctx_u, ctx)
        i = np.clip(i, 0, len(self.ctx_u) - 1)
        seen = self.ctx_u[i] == ctx
        tot = np.where(seen, self.ctx_tot[i], 0)
        key = ctx * self.V + nxt
        j = np.clip(np.searchsorted(self.u, key), 0, len(self.u) - 1)
        cnt = np.where(self.u[j] == key, self.c[j], 0)
        p = (cnt + ALPHA) / (tot + ALPHA * self.V)
        top1 = np.where(seen, self.best[i] == nxt, False)
        return p, seen, top1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", default="dw-8ray")
    ap.add_argument("--max-order", type=int, default=6,
                   help="contexts are base-V int64 codes, so V**(k+1) must fit in int64: k <= 6 at V=422")
    a = ap.parse_args()
    t0 = time.time()
    tdir = REPO / "datasets" / "discworld" / a.instance / "tokens"
    tok, _, vocab, meta = load_tokens(tdir)
    V = vocab.size
    if V ** (a.max_order + 1) >= 2 ** 63:
        raise SystemExit(f"--max-order {a.max_order} overflows the int64 context key at V={V}")
    test = np.load(tdir / "test.npy").astype(np.int64)
    n, T = test.shape
    orders = {}
    for k in range(1, a.max_order + 1):
        u, c = ngram_counts(tok, k, V)
        orders[k] = Order(k, u, c, V)
        print(f"  order {k}: {len(u):,} (ctx, next) pairs over {len(orders[k].ctx_u):,} contexts  "
              f"[{(time.time() - t0) / 60:.1f} min]", flush=True)
    # score every test position m = 1..T-1 (predict frame m from frames < m), backing off
    res = {"instance": a.instance, "V": int(V), "n_test": int(n), "alpha": ALPHA, "orders": {}}
    unigram = np.bincount(np.asarray(tok[:200_000]).ravel().astype(np.int64), minlength=V) + ALPHA
    unigram = unigram / unigram.sum()
    for K in range(1, a.max_order + 1):
        p_all = np.zeros((n, T - 1))
        top_all = np.zeros((n, T - 1), bool)
        used = np.zeros((n, T - 1), np.int64)
        done = np.zeros((n, T - 1), bool)
        for k in range(K, 0, -1):
            for m in range(k, T):
                col = m - 1
                if done[:, col].all():
                    continue
                ctx = np.zeros(n, np.int64)
                for j in range(k):
                    ctx = ctx * V + test[:, m - k + j]
                p, seen, top1 = orders[k].lookup(ctx, test[:, m])
                take = seen & ~done[:, col]
                p_all[take, col], top_all[take, col], used[take, col] = p[take], top1[take], k
                done[take, col] = True
        # positions never seen at any order: the unigram
        p_all[~done] = unigram[test[:, 1:][~done]]
        ce = -np.log(p_all)
        by_pos = ce.mean(0)
        res["orders"][K] = {"ce": float(ce.mean()), "top1": float(top_all.mean()),
                            "ce_t_ge_5": float(ce[:, 5:].mean()), "top1_t_ge_5": float(top_all[:, 5:].mean()),
                            "ce_by_position": by_pos.tolist(),
                            "share_backed_off": float((used < K).mean()),
                            "n_contexts": int(len(orders[K].ctx_u))}
        print(f"  backoff n-gram K={K}: CE {ce.mean():.4f}  top-1 {top_all.mean():.4f}  "
              f"(t>=5: CE {ce[:, 5:].mean():.4f} top-1 {top_all[:, 5:].mean():.4f})  "
              f"backed off {(used < K).mean():.3f}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    out = EXP / "scores" / f"ngram_floor_{a.instance}.json"
    out.write_text(json.dumps(res, indent=1))
    print("done", out, f"[{res['minutes']} min]")


if __name__ == "__main__":
    main()
