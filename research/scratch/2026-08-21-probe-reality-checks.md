# 2026-08-21 — Four controls on what our probes are actually reading

Scripts in `notebooks/experiments/editability/othello_transfer/`: `probe_scaling.py`,
`obs_window_probe.py`, `random_init_control.py`, `occlusion_probe.py`. Model `W16`
(`runs/transformers/W16`), dataset `4_fixed_refl_inview`, residual point 3 unless noted, the
thread's own probe and fitting loop (`othello_probe.fit_probe`, hidden 512, 200 epochs, held out
by sequence). ~5 min total. **No models trained.**

Motivation: `othello_transfer` (2026-08-20) cleared our *editor implementation*. These four ask
the prior question — **is our probe reading a learned state at all, or is it reading the
observation?** — because if it is the latter, the whole editability framing is weaker than it
looks. Prompted by Sevan raising the low-dimensional-observation-manifold hypothesis.

## 1. Probe training data — not a confound

60x more probe data (48k → 2.88M rows) moves MLP position R² **0.9315 → 0.9604**; successive
steps +0.015 / +0.007 / +0.006 / **+0.001**. Saturated by ~1.5M rows, short of Li et al.'s ~6.7M.
Test-split fit at 1500 sequences reproduces the published **0.9349** exactly. Full table in
`2026-08-21-probe-data-scaling.md`. **The ~140x probe-data gap against Li et al. explains nothing.**

## 2. Observation-window baseline — how much is temporal integration?

Linear probe on the raw (noisy) observation the model actually receives, sweeping window length,
frames t ≥ 15, 5000 sequences:

| window | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| linear R² | 0.292 | 0.305 | 0.313 | 0.319 | **0.323** |
| MLP R² | 0.837 | 0.866 | **0.878** | 0.864 | 0.851 |

(A 40-frame window is degenerate — only one timestep qualifies, 4,000 rows against 5,120 input
dims — and is not reported.) **Temporal integration buys almost nothing linearly: +0.03 over 16
frames.** The 2026-08-05 single-frame `clean_obs` numbers (linear 0.259 / MLP 0.754) are
consistent with the window-1 column here.

## 3. Random-weight baseline — how much is the architecture?

Li et al.'s own control (their `--random` arm). Same architecture, data, probe, split; only the
weights change. Two random seeds, averaged (they agree to ±0.01):

| residual point | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| trained, linear | 0.594 | 0.778 | **0.803** | 0.766 | 0.755 |
| random, linear | 0.468 | 0.567 | 0.559 | 0.547 | 0.536 |
| trained, MLP | 0.853 | 0.917 | 0.930 | 0.942 | **0.944** |
| random, MLP | 0.812 | **0.842** | 0.819 | 0.802 | 0.785 |

**⚠ This corrects an overstatement made earlier the same day.** Against the raw observation the
trained latent gains +0.44 linear R², and that was reported as the model's contribution. Against a
random-init network at the same point the gain is **+0.244** — roughly half of it is the
architecture, not learning.

What survives, and it is still substantial:
- **Training contributes +0.244 linear and +0.14 MLP** over random init.
- **The depth trend is the cleaner signature.** Random decodability *declines* with depth
  (linear 0.567 → 0.536, MLP 0.842 → 0.785); trained decodability *rises and plateaus*
  (linear 0.594 → 0.803, MLP 0.853 → 0.944). Random features degrade as they are mixed; learned
  features accumulate. No random projection reproduces that.
- **The model's achievement is linearisation.** Nonlinear position information is already in the
  raw observation (MLP 0.851); the trained latent adds only +0.09. Linearly the observation gives
  0.323 and the trained latent 0.803. The model is not discovering position — it is making
  position *linearly accessible*.

**The uncomfortable reading, stated plainly:** nonlinearly, our probe is mostly reading the
observation, not a learned state. Contrast Li et al., whose randomized network sits at ~26% error
against a 47% constant-guess floor while their trained nonlinear probe reaches **1.7%** — training
removes ~93% of the random model's error, against ~71% of the random model's unexplained variance
here. **In their world board-state decodability is almost entirely computed by the model; in ours
most of the nonlinear position decodability is supplied by the observation.** That is a structural
difference between the two settings, on exactly the axis the planned architecture run is meant to
isolate.

## 4. Occluded discs — the control that does not work, and why

Sevan's proposal: restrict the analysis to discs the observation *cannot* show. Occlusion defined
from `obs_id` (zero rays contributed), **not** `is_visible` — see the trap below. 10,000 sequences,
t ≥ 15, 15,110 occluded instances (3.02%); mean-predictor RMSE ≈ 1.83 sim units.

| representation | obj 0 visible | obj 0 occluded | obj 1 visible | obj 1 occluded |
|---|---|---|---|---|
| obs, 1 frame (MLP) | 0.815 | 0.976 | 0.492 | 0.868 |
| obs, 16-frame window (MLP) | 0.715 | 1.001 | 0.451 | 0.801 |
| random-init latent (MLP) | 0.828 | 0.984 | 0.504 | **0.773** |
| **trained latent (MLP)** | **0.422** | **0.812** | **0.287** | 0.779 |

**A single frame reads a fully occluded disc at RMSE 0.87–0.98, far better than chance.** That is
only possible because **occlusion in this world is a strong positional constraint**: a hidden disc
must lie behind the visible one, inside its angular shadow. "Occluded" here means *localised to a
cone*, not *unobserved* — so the control cannot separate carried state from present evidence.

Consequently: on object 1 the trained latent (0.779) is **no better than random init** (0.773) or
the raw window (0.801); only object 0 shows a gain (0.812 vs 0.976–1.001, 17%). And there is
**no decay with time hidden** — obj 0: 0.791 at 1 frame → 0.824 at 9+; obj 1: 0.730 → 0.893. A
dead-reckoned state should degrade; flat is what a per-frame shadow constraint predicts.

**Verdict: closer to a null than to persistence evidence, and the dataset cannot distinguish the
two.** The clean test needs discs that genuinely leave the view.

### Two traps found on the way

- **`is_visible` is a no-op on every current dataset.** It means "object overlaps the frustum", and
  `always_in_frustum=True` makes it identically True — 100% on `4_fixed_refl_inview`. The masks in
  `othello_gpt/pipeline.probe_table` and downstream have never filtered anything. Harmless to past
  results (masking nothing equals no mask) but the field is not an occlusion signal.
- **Object index is confounded with brightness.** `fixed_reflectivities=True` assigns the same
  ordered reflectivities every sample, so object 1 is decodable everywhere better than object 0
  (visible MLP 0.287 vs 0.422). 2026-08-05 found the same ("a linear map keys on brightness").
  Any per-object claim inherits this.

## Open / next

- **The clean persistence test needs a world where objects exit the frustum.**
  `1_fixed_refl_train` predates the `always_in_frustum` flag and has genuine exits, but `W16` is
  trained on dataset 4, so evaluating it there is out of distribution and invalid. Either train on
  such a world, or generate the planned run-A corpus with `always_in_frustum=False` — which buys
  the persistence test but breaks comparability with every existing editability number. **Raised
  with Sevan as a run-A decision.**
- Frustum-coordinate (angle, depth) probe targets — the discworld analogue of Nanda's mine/theirs
  frame — still untested, and may also be where velocity becomes decodable.
