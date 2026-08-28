# Discworld at scale — is our editability negative data-limited?

> ## ⛔ TRIGGER EVALUATED 2026-08-22: it does NOT fire
> The `ours_on_othello` ladder is in. Across 222× more data, absolute post-edit Edit Index moved
> **+0.059 → +0.098**, and *every* best-arm edit made **both** Li metrics worse (`D_w16`: Li error
> 3.025 → 5.367 against their model's 2.723 → 0.052). Editability does not emerge with data volume
> in a world where their architecture achieves it with the same code. Per the decision table below,
> **do not run this for editability reasons.** The 10× discworld rungs queued on 2026-08-21 still
> ran — they were nearly free and answer a separate question about discworld prediction quality —
> but the editability motivation is withdrawn. The live variable is **architecture**; see
> `our-architecture-on-othello.md` and the run-A pilot.

**Status:** `superseded — trigger did not fire`, 2026-08-22. Originally `partially running`, 2026-08-21. Sevan's proposal, made while the `ours_on_othello`
ladder was running, then upgraded the same evening: *"lean toward running the discworld data
scaling experiments because it's close to free, I'll be asleep anyway."*

> ### What is actually queued tonight, and what is not
> **Running unattended:** corpus `datasets/17_scale_900k` (900k train / 20k val / 20k test / 10k
> edits, seed base 3,000,000 — disjoint from dataset 4), and two **fixed-compute** training rungs
> chained behind the Othello ladder by `runs/ours_on_othello/queue_discworld.sh`:
> `S0c_90k` (90k prefix, 300 epochs) and `S1_900k` (900k, 30 epochs) — ~95,100 steps each.
>
> **Not queued:** the **editability evaluation**. `controls_lib.py` hardcodes dataset 4's path and
> `pinv_alpha_discworld.py` hardcodes `W16`, so pointing the probe and editor suite at a new
> checkpoint + dataset is real wiring. Unattended integration work is where errors hide, so it is
> deliberately left for a supervised session. By morning there are trained checkpoints and val
> curves, not editability numbers.
>
> **Also not attempted:** the 50× and 100× rungs. See the revised sizing below — they need a
> streaming loader, which is the one piece of new machinery this brief requires.
**Tag:** `[in-frame]` · **Sub-Q:** 3

## The question

Every discworld world model in this repo is trained on **90,000 episodes × 40 frames = 3.6M unique
frames**. Every editability negative we have rests on models trained at that volume. Nobody has
ever asked whether the negative survives more data, **holding the architecture fixed**.

This is the one-variable version of the confound that
[`othello-architecture-on-discworld.md`](othello-architecture-on-discworld.md) (run A) attacks with
two variables at once. Run A changes architecture *and* data scale deliberately; this changes only
data scale. If it turns out to be the informative one, it is also far cheaper.

## ⛔ The trigger — the reason this is queued rather than proposed

Sevan, 2026-08-21: *"that makes much more sense specifically if ours on othello starts being
editable but only after more training data."*

**Run this only if the `ours_on_othello` scale ladder shows editability emerging with data volume**
— that is, if Edit Index gain over each model's own null rises materially from `M` (90k games)
through `L1` / `L2` / `D` (20M, fixed compute), in a world where their architecture is already
known to be editable.

| ladder outcome | what to do |
|---|---|
| **gain rises with data** — e.g. M +0.09 → D ≫ +0.5 | **run this.** Data volume is a live cause of the editability negative, and testing it in our own world is the direct follow-up |
| **gain stays flat and small at every rung** | **do not run this.** Our architecture fails to become editable even at 222× the data in a world where editability is demonstrably achievable, so more discworld data would not be expected to help. Run A (their architecture) becomes the priority instead |
| **gates never pass at any rung** (legal-move mass stays < 0.95 through `D`/`F`) | the ladder is uninformative about editability; neither branch is licensed. Diagnose capacity/depth first — see the note below |

The 2026-08-21 evidence that motivates the trigger: at 90k games our architecture reaches only
**0.73–0.76 held-out legal-move mass** on Othello (Li et al.: 0.9998), and **even after memorising
the training set** — training CE 1.183 against a Bayes floor of 2.009 — training legal mass reaches
only **0.863**. Memorisation and rule-learning are different targets, and this architecture at this
data volume has done the former and not the latter.

## Design, if the trigger fires

**One variable.** Same architecture (`W16`: `d_model` 256, 4 layers, 4 heads, RoPE, band 16), same
objective (MSE next-observation), same optimiser and schedule, same dataset *configuration*
(`4_fixed_refl_inview`: 2 objects, 40 frames, `obs_res` 128, obs noise 0.2, position noise 0.04,
`always_in_frustum` true). **Only the number of unique episodes changes.**

Mirror the ladder that made `ours_on_othello` interpretable — **fixed optimiser steps, varying
pool** — so that anything that moves is diversity rather than compute:

| rung | unique episodes | unique frames | vs today's 3.6M | steps |
|---|---|---|---|---|
| `S0` | 90,000 | 3.6M | 1× (**today's `W16`, the anchor**) | 95,100 |
| `S1` | 900,000 | 36M | 10× | 95,100 |
| `S2` | 4,500,000 | 180M | 50× | 95,100 |
| `S3` | 9,000,000 | 360M | 100× | 95,100 |

`S0` must reproduce `W16`'s published val loss (0.02359) and position probe R² (0.798 linear /
0.9349 MLP), or the port is broken and nothing downstream is interpretable.

**Feasibility, re-measured 2026-08-21 — the earlier estimate was wrong.** I had quoted "645
episodes/s/core"; the measured rate is **~1,450 episodes/s on 8 workers**, i.e. ~180/s/core, and
the per-episode footprint from the real dataset is **47.8 KB all fields / 20.4 KB `obs_intensity`
alone**. (Same per-core-vs-total error I made on the Othello generator the same day. Measure it.)

| rung | episodes | × today | generation | disk, all fields | `obs_intensity` fp32 | fits? |
|---|---|---|---|---|---|---|
| `S0` | 90,000 | 1× | 1 min | 4 GB | 2 GB | ✓ — **and it already exists as `W16`** |
| `S1` | 900,000 | 10× | ~11 min | 43 GB | **18.4 GB** | ✓ resident on a 32 GB card |
| `S2` | 4,500,000 | 50× | ~55 min | 215 GB | 92 GB | ✗ needs a streaming loader |
| `S3` | 9,000,000 | 100× | ~110 min | 430 GB | 184 GB | ✗ needs a streaming loader |

⛔ **The binding constraint is `build_inmemory_dataloaders`, which puts the whole `obs_intensity`
array on the GPU.** That caps an unmodified run at ~1.2M episodes on a 32 GB card. Beyond that,
either add a memmap path or store `obs_intensity` as **uint8** (intensities are in [0,1] against
observation noise σ = 0.2, so 1/255 resolution is ample) — which brings 4.5M to 23 GB and 9M to
46 GB. Do this in a thread-local loader, **not** by changing the shared `pim/world_models/dataloader.py`,
which other threads depend on. `scripts/train_transformer.py` gained a backward-compatible
`--limit` flag on 2026-08-21 so a smaller rung is a strict prefix of a larger corpus.

⚠ **Seed ranges must be disjoint from `datasets/4_fixed_refl_inview`'s test and edits splits**, and
asserted, not assumed. This trap is already pinned in run A's brief and was enforced by hashing
token rows in `ours_on_othello/corpus.py`; reuse that pattern.

## Metrics — all existing, none new

Position/velocity probe R² by residual point against the **random-init** floor
(`othello_transfer/controls_lib.py`), and the canonical §4 editing suite
(`notebooks/experiments/editability/METRICS_AND_EDITORS.md`): Edit Index union, fidelity ratio,
target / ghost / collateral RMSE, plus a waterfall through `waterfall_grid(...)` for any claim
about generations.

⛔ **Single-point *and* all-point pseudoinverse writes, with the step size swept.** 2026-08-21
established that on Othello this editor only works at a single mid-depth point and that α matters
by ~50×; `pinv_alpha_discworld.py` already ran that sweep on `W16` and found it inert at every
(point, α). Repeat it at each rung — the question is whether *data* changes that.

## Decision rule

Editability gain over each rung's own unsteered Edit Index, plotted against unique episodes.
**Rises materially** → the discworld negative is data-limited and every prior negative in the
thread needs the scope qualifier "at 3.6M frames". **Flat** → the negative survives a 100× data
increase at fixed architecture and compute, which is a much stronger statement than anything we
currently have, and closes the data confound for good.

Either way this is worth having: it is the only experiment that puts a *scope* on the thread's
central negative.

## Ordering

After `ours_on_othello` completes and its trigger is evaluated. Before run A if the trigger fires
(one variable beats two); after run A, or not at all, if it does not.
