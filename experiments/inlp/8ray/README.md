# inlp_8ray — the nullspace cascade on the 8-ray transformer

**Question** (2026-09-04): on `L-dw-8ray-20m` (dw-8ray: radius 1.0, 8 usable rays), how large
is the linearly readable code — per residual point and **per state component** — and does
writing all of it edit any better than the canonical single-probe PI (+0.30)?

**What runs** — `drivers/inlp_8ray.sh` calls the canonical INLP experiment script
(`experiments/inlp/scripts/inlp_dw.py`, extended 2026-09-04 with a per-component held-out
R² profile per cascade probe and an optional scratch cache dir for smokes) on the run,
frustum then cartesian, `n_seq` 20,000 (the same setting as the L-dw-20m INLP run).
Cascades are persisted into `runs/ray_ablation/L-dw-8ray-20m/probes/` via ProbeCache.
Unit `inlp_8ray` (MemoryMax 30G), logs in `logs/inlp/8ray/`.

**Outputs** — `scores/inlp_L-dw-8ray-20m_{frustum,cartesian}.json`: per point the number of
probes, total rank, aggregate and per-component R² profiles, the K × α edit sweep with
Edit Index + fidelity, and the K=1 wiring check against the canonical PI step.

**Result (2026-09-04)** — 29–40 orthogonal probes per point (rank 232–320; L-dw-20m had 15–20),
aggregate R² decaying only 0.95 → 0.83 over the first eight; by component the redundancy is
mostly position (still R² > 0.5 at probe 15; velocity halves by probe ~10, < 0.2 by ~14). Writing K=8–16 probes at once
lifts the best Edit Index from +0.297 to +0.358 (frustum) / +0.242 to +0.336 (cartesian) at
fidelity ≈ 1.1; writing every probe is no better. Wiring check 6–10 %; probe-1 per-component
R² = Table 1b. Cascades persisted in the run's `probes/`. Write-up:
`research/findings/inlp-8ray.md`.
