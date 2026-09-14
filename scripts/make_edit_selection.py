#!/usr/bin/env python
"""Write a discworld instance's canonical edit-case SELECTION — the bench every scorer uses.

    .pim/bin/python scripts/make_edit_selection.py --instance dw-noiseless
    .pim/bin/python scripts/make_edit_selection.py --instance dw-8ray --pool 6000

The rule (2026-09-12, every instance, one rule): the first ``--n`` cases, in the edits split's
own order, whose two clean renders at the edit frame differ on at least ``--min-rays`` rays.
That excludes a teleport that renders an identical frame (20 % of cases at 8 rays, 0.5 % at
128) and, on dw-blink, every case whose edited object is hidden at the edit frame (no
differing rays) — the blackout population has its own analysis. Where the instance has a
token vocabulary (dw-8ray) both frames and the whole context must also be in it, so the
frame model and the token model score the SAME cases.

Writes ``edits/v1/selection.json`` (``pim.environments.layout.edits_selection``); an existing
file is moved to ``_unused/`` with a timestamp, never overwritten in place. Reads through
``pim.environments.discworld.bench.bench_arrays(use_selection=False)`` so the rule and the
bench cannot drift apart. Generalises experiments/interface_ablation/edits_audit (2026-09-08).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments import layout  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--pool", type=int, default=4000, help="cases scanned, in split order")
    ap.add_argument("--min-rays", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    arr = dwb.bench_arrays(n=a.pool, target="full", basis_name="frustum", instance=a.instance,
                           use_selection=False)
    pre_f, post_f = arr["zones"].gt_unedited, arr["clean"][:, EF]
    nray = (np.abs(pre_f - post_f) > 1e-6).sum(1)
    ok = nray >= a.min_rays
    stats = {"pool": int(len(ok)), "pool_identical_frame": int((nray == 0).sum()),
             "pool_ge_min_rays": int(ok.sum())}
    vocab_p = layout.tokens_dir(a.instance) / "vocab.npz"
    if vocab_p.exists():
        from pim.environments.discworld.tokens import UNK, FrameVocab, encode

        vocab = FrameVocab.load(vocab_p)
        pre, post = encode(pre_f, vocab).astype(int), encode(post_f, vocab).astype(int)
        tok = encode(arr["obs"][:, :EF], vocab).astype(int)
        in_vocab = (pre != UNK) & (post != UNK) & (tok != UNK).all(1)
        stats["pool_in_vocab"] = int(in_vocab.sum())
        ok &= in_vocab
    sel = np.where(ok)[0][: a.n]
    if len(sel) < a.n:
        raise SystemExit(f"only {len(sel)} valid cases in a pool of {a.pool}; raise --pool")
    i = np.arange(len(pre_f))
    tele = np.linalg.norm(arr["pos"][i, EF, arr["edit_object"]] - arr["pos"][i, EF - 1, arr["edit_object"]], axis=-1)
    out = {"instance": a.instance, "created": time.strftime("%Y-%m-%d %H:%M"), "n": int(len(sel)),
           "pool": a.pool, "min_rays": a.min_rays, "vocab": str(vocab_p) if vocab_p.exists() else None,
           "rule": f"the FIRST {a.n} cases in split order whose two clean renders differ on >= {a.min_rays} rays"
                   + (", both frames and the context in the token vocabulary" if vocab_p.exists() else ""),
           "select": sel.tolist(),
           "stats": {**stats, "scanned": int(sel[-1]) + 1,
                     "teleport_mean_selected": float(tele[sel].mean()), "teleport_mean_pool": float(tele.mean()),
                     "rays_changed_selected": {int(k): int(v) for k, v in zip(*np.unique(nray[sel], return_counts=True))}}}
    p = layout.edits_selection("discworld", a.instance)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = layout.unused_dir("discworld", a.instance) / "edits_v1" / f"selection.{time.strftime('%Y-%m-%d')}.pre.json"
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, old)
        print(f"previous selection moved -> {old.relative_to(_REPO)}")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "select"}, indent=1))
    print(f"-> {p.relative_to(_REPO)}  ({len(sel)} cases, {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
