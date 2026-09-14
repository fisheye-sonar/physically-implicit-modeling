#!/usr/bin/env python3
"""Canonical scores of the dropout arms side by side, read from each run's scores.json (master_eval output;
no metric math here). Prints eval version, gates, probe skill and — per editor — the best arm by union Edit
Index (master_eval's `best`), the same arm's symmetric-difference index, and the best FIDELITY-GUARDED arm
(fidelity ratio <= 1.1, the guard used in the INLP K-copy tables).

    python experiments/dropout_ablation/scripts/canonical_table.py "label=runs/topic/run" ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def best(arms, editor, guard=None, key="edit_index_union"):
    xs = [a for a in arms if a["editor"] == editor and (guard is None or a["fidelity_ratio"] <= guard)]
    return max(xs, key=lambda a: a[key]) if xs else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+", help='"label=runs/<topic>/<run>"'); a = ap.parse_args()
    rows = []
    for spec in a.runs:
        label, path = spec.split("=", 1); s = json.loads((REPO / path / "scores.json").read_text()); rows.append((label, s))
    print(f"{'run':22s} {'eval':13s} {'val':>7s} {'CE':>7s} {'Bayes':>7s} {'legal':>6s} {'LIN':>6s} {'MLP':>6s} {'uned U':>7s} {'uned SD':>8s}")
    for label, s in rows:
        g = s["gates"]; u = s["unedited"]
        print(f"{label:22s} {s['eval_version']:13s} {s['val_loss']:7.4f} {g['ce']:7.4f} {g['bayes_ce']:7.4f} {g['legal_mass']:6.3f} "
              f"{max(s['probe_skill']['mine|linear|sequence']):6.3f} {max(s['probe_skill']['mine|mlp|sequence']):6.3f} {u['edit_index_union']:+7.3f} {u['edit_index_symdiff']:+8.3f}")
    for ed in ("PI", "ND", "GS"):
        print(f"\n{ed}: best by union EI (unguarded)  → union / symdiff / fid (point, α)      | best GUARDED (fid ≤ 1.1) → union / symdiff / fid (point, α)")
        for label, s in rows:
            b = best(s["arms"], ed); gd = best(s["arms"], ed, guard=1.1)
            fb = f"{b['edit_index_union']:+.3f} / {b['edit_index_symdiff']:+.3f} / {b['fidelity_ratio']:.2f} (pt {b['point']}, α{b['alpha']:g})" if b else "—"
            fg = f"{gd['edit_index_union']:+.3f} / {gd['edit_index_symdiff']:+.3f} / {gd['fidelity_ratio']:.2f} (pt {gd['point']}, α{gd['alpha']:g})" if gd else "no guarded arm"
            print(f"  {label:22s} {fb:44s} | {fg}")


if __name__ == "__main__":
    main()
