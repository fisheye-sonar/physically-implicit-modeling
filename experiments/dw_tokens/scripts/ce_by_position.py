#!/usr/bin/env python
"""Where does the token model's cross-entropy sit along the sequence?

A noiseless world is deterministic once the state is identifiable from the history, so the
next-frame entropy should collapse to ~0 after the first few frames; whatever CE remains
later is model error, not irreducible uncertainty. This reads the held-out TEST split
(10k sequences) and reports, per input position t (predicting frame t+1): mean CE, top-1
accuracy, and the mean probability on the true next frame, plus the same numbers for the
"frame repeats" baseline (predict frame t again) — how much of the task is persistence.

Output: experiments/dw_tokens/scores/ce_by_position_<run>.json (+ a printed table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint  # noqa: E402

EXP = REPO / "experiments" / "dw_tokens"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/interface_ablation/L-dw-8ray-tok-20m")
    ap.add_argument("--instance", default="dw-8ray")
    a = ap.parse_args()
    run_dir = (REPO / a.run).resolve()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = load_checkpoint(run_dir / "best_model.pt", device=dev)
    model.eval()
    tok = np.load(REPO / "datasets" / "discworld" / a.instance / "tokens" / "test.npy")
    T = tok.shape[1]
    block = T - 1
    ce, acc, ptrue, persist = (np.zeros(block) for _ in range(4))
    n = 0
    with torch.no_grad():
        for i in range(0, len(tok), 500):
            x = torch.from_numpy(tok[i: i + 500]).long().to(dev)
            lg = model.logits(x[:, :block])                     # (B, block, V)
            y = x[:, 1: block + 1]
            lp = F.log_softmax(lg.float(), -1)
            ce += (-lp.gather(-1, y[..., None])[..., 0]).sum(0).cpu().numpy()
            ptrue += lp.gather(-1, y[..., None])[..., 0].exp().sum(0).cpu().numpy()
            acc += (lg.argmax(-1) == y).float().sum(0).cpu().numpy()
            persist += (x[:, :block] == y).float().sum(0).cpu().numpy()
            n += len(x)
    res = {"run": run_dir.name, "n_sequences": int(n), "block": int(block),
           "ce_by_position": (ce / n).tolist(), "top1_by_position": (acc / n).tolist(),
           "p_true_by_position": (ptrue / n).tolist(),
           "persistence_top1_by_position": (persist / n).tolist(),
           "ce_mean": float(ce.sum() / (n * block)), "top1_mean": float(acc.sum() / (n * block)),
           "persistence_top1_mean": float(persist.sum() / (n * block)),
           "ce_mean_t_ge_5": float(ce[5:].sum() / (n * (block - 5))),
           "top1_mean_t_ge_5": float(acc[5:].sum() / (n * (block - 5)))}
    (EXP / "scores").mkdir(parents=True, exist_ok=True)
    out = EXP / "scores" / f"ce_by_position_{run_dir.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"{run_dir.name}: {n:,} test sequences · mean CE {res['ce_mean']:.4f} · top-1 {res['top1_mean']:.4f} "
          f"· persistence top-1 {res['persistence_top1_mean']:.4f} · t>=5: CE {res['ce_mean_t_ge_5']:.4f} top-1 {res['top1_mean_t_ge_5']:.4f}")
    print(f"{'t':>3} {'CE':>8} {'top1':>7} {'p_true':>7} {'persist':>8}")
    for t in range(block):
        print(f"{t:>3} {ce[t] / n:>8.4f} {acc[t] / n:>7.4f} {ptrue[t] / n:>7.4f} {persist[t] / n:>8.4f}")
    print("done", out)


if __name__ == "__main__":
    main()
