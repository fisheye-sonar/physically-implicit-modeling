# ray_ablation — analyses around the 8-ray, radius-1.0 instance (runs/ray_ablation/*)

Consolidated 2026-09-06 (depth over breadth): `alpha_check/` — the PI α-grid edge check and
the unique-frame count on the 20M dw-8ray corpus (421 distinct frames); `waterfalls/` — the
ray-count waterfall panels for L-dw-8ray-20m. The instance's INLP cascades live in
`experiments/inlp/8ray/`, its tokenised twin in `experiments/dw_tokens/`. Finding:
`research/findings/ray-ablation.md`; chain logs `logs/ray_ablation/`.

**2026-09-15 — dw-16ray.** `runs/ray_ablation/L-dw-16ray-20m` (built and trained on the WSL remote by `scripts/drivers/dw_16ray.sh`;
logs `logs/ray_ablation/dw_16ray/`, `logs/dw_16ray_fac/`): the ray axis upward from dw-8ray. Finding: `probe-target-type.md`
§The ray axis upward; `ray-ablation.md` addendum. The 51 GB training corpus stays on the remote; everything else is here.
