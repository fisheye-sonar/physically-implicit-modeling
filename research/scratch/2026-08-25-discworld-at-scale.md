# Discworld at 20M: the model reaches the Bayes floor and is still not editable

**2026-08-24 → 08-25.** The matched counterpart to `BIG20M_othello_L`. Same architecture, same
optimiser, same schedule, same step count — only the environment changes.

Run `runs/discworld_scale/BIG20M_discworld_L` · corpus `datasets/20_dwscale_20m` ·
probe corpus `datasets/21_dwscale_probe` · code `notebooks/experiments/editability/discworld_scale/`

---

## 1. The run

| | |
|---|---|
| Transformer L | 25,371,776 params (8 layers / 8 heads / 512, dropout 0.1) |
| data | **20,000,000 sequences**, `position_noise_std=0.04` (matched to dset 4) |
| schedule | 780,000 steps = **11.09 epochs**, batch 256, 2k warmup then **constant** lr 1e-3, AdamW wd 1e-4, grad clip 1.0, seed 0 |
| result | **best val 0.022873 @ step 660,000** (epoch 9.39), 8.04 h |
| checkpoints | **22** — log-spaced ∪ per-epoch |

Every one of those settings is identical to the Othello 20M run. What cannot match, because it
*is* the environment: block 39 vs 59, `Linear(128,512)`/`Linear(512,128)` vs
`Embedding(61,512)`/`Linear(512,61)`, MSE vs cross-entropy.

**It plateaued at ~step 50,000 — 6% of the run.**

| step | val |
|---|---|
| 5,000 | 0.023660 |
| 50,000 | 0.022978 |
| 135,000 | 0.022955 |
| 280,000 | 0.022917 |
| 475,000 | 0.022915 |
| **660,000** | **0.022873** |
| 780,000 | 0.022897 |

610,000 steps and 6.7 GPU-hours after step 50,000 bought **0.000105**. The train–val gap sits at
**~0 throughout** — it is not overfitting that stops it. Contrast Othello, still setting new bests
at step 775,000 of 780,000.

---

## 2. Why it plateaus: it is at the Bayes floor

Measured on 400 sequences × 39 frames of the probe corpus, scored exactly the way `val_loss` is
(MSE against the **noisy** observation, all 39 predicted frames). Script kept at
`scratchpad/floor.py`; method below so it can be rebuilt.

| bound | MSE | RMSE |
|---|---|---|
| **A.** knows `clean_{t+1}` exactly — obs-noise floor | 0.018866 | 0.1374 |
| **B.** knows true `(pos, vel)` at *t*, integrates process noise — **strict lower bound** | 0.022171 | 0.1489 |
| the model | 0.022873 | 0.1512 |

**Decomposition of the model's loss:**

| component | MSE | share |
|---|---|---|
| observation noise on the target | 0.018866 | **82.5%** |
| process noise (next position genuinely undetermined) | 0.003305 | **14.4%** |
| state inference + any model shortfall | 0.000702 | **3.1%** |

**The model is 3.16% above a state-omniscient oracle.** True Bayes risk is bracketed
**[0.022171, 0.022873]** — below by the oracle (the state is a sufficient statistic, so it has
strictly more information than any observation history), above by the model itself (a concrete
predictor upper-bounds the Bayes risk). **Maximum possible gain from any further training:
0.000702 MSE, 1.5% in RMSE, and truthfully less.**

### How the bounds are computed

Both come from the simulator's own generative process, nothing fitted:

```
p_{t+1} = p_t + v_t·dt + N(0, 0.04²)          process noise
clean_t = render(p_t)                          deterministic ray-cast
obs_t   = clip(clean_t + N(0, 0.2²), 0, 1)     observation noise, CLIPPED
```

The one closed form is the **censored-normal mean** — the optimal prediction of the *noisy*
observation given a clean value `c`:

    g(c) = σ[φ(a) − φ(b)] + c[Φ(b) − Φ(a)] + (1 − Φ(b)),   a = −c/σ,  b = (1−c)/σ

Sanity: `g(0) = σ/√(2π) = 0.0798`; background rays measure **0.0799**. (This is also why the
Bayes-optimal background is *not* pure black — see §6.)

* **A** uses `clean` recovered losslessly via `reconstruct_clean_obs(obs_id, reflectivities)`:
  `mean (g(clean_{t+1}) − obs_{t+1})²`.
* **B** takes the dataset's true `pos[t], vel[t]` and Monte-Carlos the process noise (64 draws):
  `pred = mean_w g(render(p_t + v_t·dt + w))`.

⚠ **Average the renders, not the positions.** `render(E[p]) ≠ E[render(p)]` — the renderer has
sharp disc silhouettes, so the correct object has *blurred* edges. Using `render(p + v·dt)`
understates the floor.

⚠ Not computed on this corpus: the **true** Bayes risk, which needs the particle filter over the
8-D state (run on dset 4 on 2026-08-24, bracketing [0.1446, 0.1564] RMSE at one mid-sequence
frame). That is why a bracket is reported and not a point estimate.

---

## 3. Editability — still nothing, in either basis

Bench is dset 4's edits split throughout, so every number stays comparable with every prior
discworld result. Probes fit on `datasets/21_dwscale_probe` (120k available, 30,000 used).
**Unedited baseline EI −0.6998** in both bases.

| editor | cartesian `pos` | cartesian `full` | **inv_y `pos`** | **inv_y `full`** |
|---|---|---|---|---|
| PI injection (1 point) | +0.0475 | +0.0522 | **+0.0874** | **+0.0854** |
| Nanda addition | −0.0380 | −0.0915 | −0.0172 | −0.0513 |
| MLP grad steering | −0.2043 | −0.1895 | *(not run — linear-only)* | — |

### PI injection is destructive, not a weak success

Its "best" arm sits at **alpha 175 — the top of the sweep**, index still climbing at the boundary.

| metric | unedited | PI (cartesian `full`) | |
|---|---|---|---|
| target RMSE ↓ | 0.4850 | 0.5899 | **1.22× worse** |
| ghost RMSE ↓ | 0.5774 | 0.3269 | 0.57× better |
| **collateral RMSE ↓** | **0.1225** | **0.6644** | **5.4× worse** |
| edit-frame RMSE ↓ | 0.2690 | 0.3852 | 1.43× worse |

The alpha sweep shows EI rising **monotonically with collateral damage** (alpha 0.1 → 175:
EI −0.698 → +0.052, collateral 0.1227 → 0.6644). The edit erases the scene: the old object leaves
(ghost improves — the easy half, any destruction does that), the new one never arrives (target
worsens), the untouched object is wrecked. The output is far from *both* references, so
`d_u ≈ d_e` and the index drifts to ~0 from below. **An Edit Index near zero is the ambiguity
point, not a partial success.**

⚠ **`fidelity_ratio` cannot see this.** It is `gt_traj_rmse(editor)/gt_traj_rmse(unsteered)`, a
whole-rollout average, and reads 0.993–0.994 — apparently fine. In the frustum basis it reads
**1.09–1.35, i.e. above 1**, meaning the edited rollout ended *further* from the true post-edit
world than doing nothing. Always read the zone-resolved metrics (target / ghost / collateral)
beside it.

For contrast, Othello's PI at +0.6104 moved both guards the right way: Li error 2.763 → 0.114,
legal mass 0.857 → 0.990.

---

## 4. Probe decodability, and the frustum basis

Linear probes, `target=full`, best residual point (pt 3 in both bases). ⚠ These are **different
quantities**, not the same number twice: cartesian is `(x, y, vx, vy)` per object; `inv_y` is
`(u, 1/y)` and derivatives, `u = x/(k·y)` being the ray coordinate.

| cartesian | R² | | frustum `inv_y` | R² |
|---|---|---|---|---|
| obj0 x | 0.9506 | | obj0 u | 0.9759 |
| obj0 y | 0.8970 | | obj0 1/y | 0.9418 |
| obj1 x | 0.9843 | | obj1 u | 0.9923 |
| obj1 y | 0.9525 | | obj1 1/y | 0.9766 |
| obj0 vx | 0.7108 | | obj0 du | 0.7340 |
| obj0 vy | 0.5260 | | obj0 d(1/y) | 0.5386 |
| obj1 vx | 0.8272 | | obj1 du | 0.8418 |
| obj1 vy | 0.6872 | | obj1 d(1/y) | 0.6847 |

**position mean 0.9461 → 0.9716 · velocity mean 0.6878 → 0.6998**

The depth coordinate gains most (`y` 0.8970 → `1/y` 0.9418), which is what the `w ∝ 1/depth`
argument predicts. **Velocity barely moves (+0.012)** — depth-rate stays the weak axis in either
parameterisation.

MLP probes (cartesian only, 30k): best 0.9750 @ pt 5; per-dim `x` 0.9922 / `y` 0.9479 /
`x` 0.9960 / `y` 0.9707 / `vx` 0.8743 / `vy` 0.5989 / `vx` 0.9089 / `vy` 0.7561. Tripwire clean
(**0/9** points with MLP below linear, worst in-sample gap +0.0154) — 30k is enough.

### ⛔ Do not quote the "overall" R²

`othello_probe._r2` pools `ss_res`/`ss_tot` over **all output dims at once**, so it is
**variance-weighted**, not a per-dimension mean:

| basis | per-dim variance | position share |
|---|---|---|
| cartesian | 3.07, 3.71, 3.03, 3.57 · 0.0032, 0.0033, 0.0032, 0.0033 | **99.90%** |
| inv_y | 0.179, 0.0015, 0.175, 0.0014 · 0.0002, ~0, 0.0002, ~0 | **99.88%** |

Velocity carries ~0.1% of total variance, so "overall" is position with velocity rounded away —
the *unweighted* mean is 0.817 for cartesian against the pooled 0.9440. Worse, within frustum
position, `u` (var 0.179) outweighs `1/y` (var 0.0015) by **120×**, so `inv_y`'s 0.9836 mostly
reports how well `u` is read and is nearly blind to the depth coordinate — the interesting one.
**Quote per-dimension numbers, or a variance-standardised R².** The fitting itself is fine
(`fit_probe` divides by `y_std`); only the summary statistic is skewed.

---

## 5. ⚠ The frustum result is UNCONTROLLED

There are **no random-init or observation-space baselines in the frustum basis**, and the
cartesian ones are on `W16`, not on this model. `obs_window_probe.py` and
`random_init_control.py` contain **zero** references to a basis parameter — they predate the
frustum work, and `2026-08-21-probe-reality-checks.md` lists frustum targets under *Open / next*.

This matters because `u = x/(k·y)` is essentially the ray index, and `always_in_frustum=True`
keeps both discs visible always — so an observation-space probe might read `u` very well with no
model involved. **The position gain 0.9461 → 0.9716 could be "the model represents depth better in
this parameterisation" or simply "these are easier functions of the raw observation".** Until the
controls are run in the frustum basis, it cannot be reported as a finding.

For reference, cartesian on W16: random-init linear 0.468–0.567, random-init MLP 0.812–0.842;
training contributed **+0.244 linear / +0.14 MLP** over random init.

Cost to fix: thread `basis_name` through both control scripts (`E._to_basis` already does the
work), then ~5 min per baseline at linear-only/30k — about **15 min** for both baselines in both
bases on this model.

---

## 6. The reference the Edit Index scores against is not Bayes-optimal

Separate finding from 2026-08-24, unchanged by this run and still unfixed.

The scorecard compares model output to the **clean render**. But the model is trained on the
**noisy** observation (`F.mse_loss(pred, obs[:, 1:, :])`), so its optimal output is
`g(clean) ≠ clean`. The gap is **+0.0798 on background** — and background is **74% of all rays**
(measured: 0.0799, with exactly 49.99% of background rays reading precisely 0). At the bright end
the clip bites the other way, −0.0169 at clean 0.8.

Consequence for the axis: a predictor that is *perfect at what it was trained to do* scores

| predictor | Edit Index |
|---|---|
| clean render of the edited world *(the current reference)* | +1.0000 |
| **Bayes-optimal prediction of the noisy observation** | **+0.8215** |
| Bayes-optimal for the unedited world | −0.8005 |
| clean render of the unedited world | −1.0000 |

**The scale is really ≈ +0.82 to −0.80, not ±1.** No conclusion to date is affected (nothing is
near the ceiling), but "+1.0 = perfect edit" is a misreading. Cheapest fix: score against
`g(clean)` on both sides — one line, no retraining, and it lifts the ceiling toward +1. Making the
*unedited* side fully correct additionally needs the particle filter, because that reference is a
**predicted** position (`pre_pos + pre_vel·dt`, noise-free) while the edited side is a **known**
one — an asymmetry in what each reference would take to be correct.

---

## 7. Where things stand

**Established.** Discworld is not editable at 20M sequences, with a model 3.16% from a
state-omniscient oracle and linear probes reading position at R² 0.95–0.97. The two easy
explanations — undertrained, or no world model — are both dead. A better-matched basis moves PI
from +0.05 to +0.09 and leaves the picture unchanged. Whatever blocks editing is about the
**geometry of the representation** or the **editors**, not model quality or data volume.

**Not done** (power outage during the analysis stage killed the chain; training had already
finished and every checkpoint survived):

1. **The 900k rung** `L90_discworld_pn04` — never started. `runs/discworld_scale/chain.sh` stage 5.
   Nested prefix of the same corpus (`--limit 900000 --steps 316406`), ~3.2 h + analysis.
2. **MLP probes / grad steering in the frustum basis** — ~8 min more on top of linear-only.
3. **Random-init and observation-space baselines in frustum** — §5, blocking.
4. The other four frustum depth candidates (`y`, `rho`, `inv_rho`, `width`). The depth coordinate
   was **never settled**: `editability-scaling-sweep.md` lists it under open decisions and the note
   it cites, `2026-08-23-frustum-basis.md`, was never written. `inv_y` was used here because it is
   `frustum.py`'s default and what its derivation argues for — **not** because it won a comparison.
   Linear-only makes all five ≈ 30 min.
