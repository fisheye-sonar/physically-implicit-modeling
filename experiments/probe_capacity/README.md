# probe_capacity

**Question:** does the random reservoir catch the trained model as the probe widens — later on
Othello than on discworld — or never? **Status:** done 2026-09-02 (1.5 h, 42 fits, all cached
in `probes/`). Widths {LIN, 16, 64, 128, 512, 1024, 2048} × {trained, random-init, observation},
one residual point per environment, on 5× the canonical probe rows (`dw-pn04/probe_250k`,
`oth-uniform/probe_large`, both additions to the manifests). Neither reservoir catches up:
discworld plateaus at 0.975 vs 0.998, Othello at 0.60 vs 0.98.

- `scripts/probe_capacity.py <env> [--smoke]` — resumable; rewrites `scores/*.json` atomically
  after every fit and re-renders `outputs/probe_capacity.png` (same function as Fig 2).
- `drivers/probe_capacity.sh` — discworld then Othello, pings, logs in `logs/probe_capacity/`.
- Finding: `research/findings/probe-capacity.md`. Figure: `build_full_table.ipynb` Fig 2
  (`pim/figures/probe_capacity.py`).
