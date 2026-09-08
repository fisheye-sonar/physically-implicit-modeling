# gs_readout_pilot — does gradient steering reach its target?

**Question** (2026-09-02): after the GS descent, how close is the probe read-out to the
requested target? Separates "the descent never converges" (a tuning problem — more
steps would fix it) from "the read-out lands and the generation ignores it" (the write
is inert to the dynamics).

**What it does** — `scripts/gs_readout_pilot.py` wraps the canonical GS hook
(`pim.editors.grad_steer.make_intervention_hook`, unchanged) with a measurement after
each layer's write: RMSE-to-target on the changed dims, fraction of the teleport covered
per edit, held-dim drift, plus the canonical hook's own loss record — and scores the
same write's rollout with the canonical scorecard (Edit Index, fidelity). Sweeps
descent length {100 (canonical), 500, 2000} × start layer {0, 4, 8} × α {0.05 … 2.0}
× dims {pos, all} × basis {frustum, cartesian} on L-dw-20m's cached MLP-128 probes
(cache hits only — nothing is fitted).

**Run** — `drivers/gs_readout_pilot.sh` (transient unit `gs_pilot`, MemoryMax 24G,
log in `logs/gs_readout_pilot/`). ~10–15 min on the local GPU.

**Results** — `scores/gs_readout_pilot_<run>.json`, atomic after every configuration.
Write-up in `research/findings/` once read.

**Result (2026-09-02, L-dw-20m)** — the descent lands (median last-layer coverage of the
teleport 0.99 at the canonical 100 steps; RMSE to target ≈ 0.001–0.007 basis units;
held dims drift ≤ 0.015), 100 → 2000 steps moves the Edit Index by +0.007 on average,
best index in the sweep +0.002 — GS is not under-converged; the generation ignores a
write the probe is satisfied by. Write-up: `research/findings/gs-readout-landing.md`.
