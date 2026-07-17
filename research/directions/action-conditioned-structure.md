# Direction: Does training on actions induce causal/editable latent structure?

**Tag:** `[reframe]` · **Sub-question:** 3 (editability) + 2 (identifiability) · **Status:** proposed ·
**Complexity:** high (new sim nudge + action-augmented dataset + action-conditioned model + retraining;
GRU primary, RSSM if cheap)

> A **standalone experiment in its own new notebook** —
> `notebooks/experiments/actions/action_conditioned_structure.ipynb`. **Do NOT touch the master
> notebook.** This one **replicates most of the master metric spread** (§1 geometry, §2 recoverability,
> §3 fiber-collapse, §4 latent editing) — for the new action-trained model vs a matched baseline.

## The question (read carefully — this reframes "actions")
Philosophically motivated (Merleau-Ponty / enactivism): **does a world model need *actions* to form a
causally disentangled representation of the world?** A purely passive observer may never individuate
"objects" as manipulable causes — which would make our editability negative result unsurprising.

**Crucial framing — we are NOT going to edit objects through the action channel at inference.** The
hypothesis is that *merely training on actions* — even **random** actions — that have a real causal
effect on the world **induces causal structure in the latent state**, such that **after training we can
discard the action channel** (hold it at no-op) and find that the *passive* latent is now more
recoverable / canonical / **editable by the existing latent editors**. So the payoff is measured on the
**passive** model with the master §1–§4 suite, especially §4 latent editability — NOT on action-driven
control. *(Completeness only: you may also check whether an edit can be produced via the action channel,
but expect it to fall short when the target displacement exceeds the small nudge range — note it, don't
optimize it.)*

## Action design (Sevan's call: discrete tokens)
Discrete action-token space aligned with how the WM literature talks about actions:
`{no-op, obj0+x, obj0−x, obj0+y, obj0−y, obj1+x, obj1−x, obj1+y, obj1−y}` (9 tokens for 2 objects).
- **Genuine no-op** must exist and be the **dominant** token (passive unfolding is the primary
  inference mode). Make non-no-op **sparse** (e.g. ~10–15% of frames) so the passive dynamics stay
  close to baseline.
- A token applies a **small persistent position nudge** to that object at that frame (offset carried
  forward from that frame — a mini position-edit, matching the affordance we care about). Magnitude
  "noticeable but small" (~0.5–1.0 world units; **smaller than a typical edit teleport**, which spans
  much of the frustum — this is why action-channel editing is expected to be limited). If a nudge would
  break frustum containment or collision, treat it as no-op that frame.
- Alignment: `a_t` perturbs the transition **into** frame `t+1`; the obs at `t+1` reflects the nudged
  world. Inject by **embedding the token and concatenating to the encoder input** (widen the encoder);
  no-op is the zero-effect token.

## Models to train (GRU; each ~9 min on this GPU — training is cheap, dataset gen is the lift)
1. **Baseline (passive, no actions).** Reuse `runs/gru/7_dset4_gru_400epochs` (dataset 4, 256 hidden,
   400 ep) — the matched dataset-4 GRU. This is the master's control condition.
2. **Action-conditioned.** Train on the **action-augmented dataset** with the token channel fed in.
   *(primary treatment)*
3. **Perturbed-passive control (REQUIRED — Sevan approved 2026-07-16).** Train on the **exact same**
   action-augmented trajectories as model 2 but **withhold the token** from the model (it sees only the
   perturbed obs — identical data, no action channel). This separates two confounds: **1→3 = perturbation
   diversity** (the world just jitters more); **3→2 = action-knowledge** (the model is *told* the cause).
   The enactivist claim predicts the **3→2** gap carries the effect, not 1→3. This control is the crux of
   the experiment — without it a positive result can't distinguish "learned agency" from "saw more diverse
   transitions." Models 2 and 3 MUST train on byte-identical trajectories (same seeds/data) for the
   comparison to be clean.

## Substrate to build (this worker owns it — it is a new pipeline)
- **Sim/dataset:** a nudge-augmented generator (extend `pim/simulator`, do NOT break existing paths):
  per-frame sample a token, apply the persistent nudge to that object, re-render. Write an `actions`
  field `(N, T)` int of token ids alongside the normal schema. Generate a train + val (and reuse dataset
  4's held-out for eval where possible). Keep obs_res=128, T=40, 2 objects, fixed reflectivities/inview
  to match dataset 4.
- **Model:** an action-conditioned GRU that **conforms to the `HiddenStateModel` protocol with actions
  defaulting to no-op**, so the *entire* master eval suite (`extractors`, `editors`, `eval`) runs
  **unchanged** on the passive model (respect Invariant 2 — no `if isinstance` branches; feed no-op
  tokens and the existing `observe_sequence`/`predict_step`/`decode`/`flat_state` must work verbatim).

## Metrics — replicate most of the master spread, PASSIVE (actions = no-op)
For all three models — baseline (1) vs perturbed-passive control (3) vs action-conditioned (2) — on the
**passive/no-op** model (the 3→2 gap is the headline; 1→3 is the perturbation-diversity control):
- **§1 Geometry:** intrinsic dim (TwoNN + MLE) and curvature of the visited-state manifold.
- **§2 Recoverability:** linear + MLP probes for `(pos, vel)`, early-t vs late-t (t<15 / t≥15).
- **§3 Fiber-collapse:** residual of best `g(pos,vel)→h`; linear→MLP drop (canonicality).
- **§4 Latent editing head-to-head:** the master editor line-up (readout injection / MLP-probe gradient
  / global-PCA / geodesic / decoder-gradient oracle) + GT(sim)/Unsteered/true-state-swap refs, with the
  obs-space selectivity/ghost/persistence metrics. **This is the headline** — is the action-trained
  latent more editable?
- **Completeness (secondary):** attempt an edit via the action channel; report reach vs a real edit target.

## Figures
Light academic theme for §1–§3 metric panels (both/all models color-coded, GT/reference columns);
dark-theme observation-space waterfalls for §4 editing (GT(sim) column, green target / red ghost). Build
every figure to hold the 2–3 models side by side. Definitions table up front. Both plots AND tables.

## Deliverables
- Executed `notebooks/experiments/actions/action_conditioned_structure.ipynb` — run **synchronously
  in-turn**, 0 error cells. PNGs → `/tmp/action_conditioned/`. New checkpoints in gitignored `runs/`.
- Dated note `research/scratch/2026-07-16-action-conditioned-structure.md`: does action-training move
  §1–§4 (esp. editability) vs baseline? Does the optional control localize the effect to
  action-knowledge vs perturbation-diversity? Mark `→ FLAG FOR PROMOTION` if signal. Do NOT touch the
  master notebook, `findings/`, or `RESEARCH.md`.

## Notes
GRU primary; RSSM only if time permits (deterministic `h` is the primary world-state carrier — report
det-only and stochastic separately). Keep the passive no-op mode as the canonical inference/eval mode.
