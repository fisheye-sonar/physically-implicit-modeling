# ray_count_waterfalls — the same scene at N = 4, 8, 16, 32 rays

**Question (2026-09-03).** In the noiseless discworld the observation is a quantisation of
a continuous state: background 0, fixed reflectivities, no anti-aliasing, so each ray is
one of {0, 0.4, 0.8} and the reachable observation set is finite — for two discs
T(N) = N(N+1) + 2·C(N+1,3) + 4·C(N+1,4) ≈ N⁴/6 (brute-force verified N = 2…10). These
animations show what that quantisation looks like as N grows.

**What is here.** `scripts/make_ray_waterfalls.py` renders ONE seed (7) of the canonical
dw-noiseless sim config at four ray counts and saves a 2D-scene + waterfall GIF per N into
`outputs/` (gitignored). Distinct frames out of 40 at seed 7: N=4 → 4, N=8 → 7, N=16 → 13,
N=32 → 23.

`--big` variant (`*_big.gif`): radius 1.0 (doubled) and speeds 0.08–0.20 (from 0.05–0.12),
same seed. Distinct frames: N=4 → 3, N=8 → 11, N=16 → 16, N=32 → 29.

Ten-example batch (`waterfall_N10_big_seed{100..109}.gif`): `--big --rays 10 --seeds 100…109`.
Radius-1.0 discs need `max_gen_attempts=5000` (the 300 default fails at seed 100) — set in the
script's SimConfig, not in canonical code. Distinct frames of 40: 14, 9, 10, 12, 9, 16, 5, 15, 7, 7.

Geometry note: the world geometry is the canonical generation geometry in every file here
(x_near 1.5, x_far 6.0, i.e. x_near = 0.25·x_far). The 0.5 in the sim is the frustum SLOPE
x/y, equal at both planes — that is what makes it a pinhole frustum matching the ray-caster's
FOV. Setting x_near = 0.5·x_far would break that match and would NOT reproduce generation.

**Status.** Illustration only; no numbers quoted anywhere durable. Pure caller of
`pim.environments.discworld` — no canonical code changed.
