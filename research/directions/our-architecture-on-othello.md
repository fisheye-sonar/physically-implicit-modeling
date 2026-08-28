# Our architecture on their world — is *our* transformer editable when the world is Othello?

**Status:** in progress, 2026-08-21 (started same day, at Sevan's request).
**Tag:** `[reframe]` · **Sub-Q:** 3
**Thread:** `notebooks/experiments/editability/ours_on_othello/`

## The question

Origin — Sevan, 2026-08-21: *"take our transformer which we trained on discworld, adapt it to match
the input and output scheme of the OthelloGPT setup, train it similar to how we trained on
discworld, and see if it is still editable. Basically applying our model on their setting instead
of the reverse as a first test, since it's cheaper."*

Two cells of a 2×2 are filled and they disagree:

| | their world (Othello) | our world (discworld) |
|---|---|---|
| **their architecture** — 8 blocks, `d_model` 512, full causal | **editable** ✓ (2026-08-20/21) | run A — [`othello-architecture-on-discworld.md`](othello-architecture-on-discworld.md), not run |
| **our architecture** — 4 blocks, `d_model` 256, RoPE, banded | **this brief** | **not editable** ✗ (2026-08-04 … 2026-08-21) |

This is the cheap cell: ~10 h against run A's ~5 days and 0.54 TB of corpus.

## Decision rule, stated before the run

- **Editable** (Edit Index gain over its own null comparable to the +1.526 our code achieves on
  their model) → our architecture and recipe are exonerated. The discworld negative is about the
  world or our data, and run A becomes largely redundant.
- **Not editable, with both gates passed** → the difference is architectural (256 vs 512 width,
  4 vs 8 layers, RoPE vs learned positions, banded vs full attention). Run A becomes the priority
  with a sharp hypothesis rather than a fishing expedition.
- **Gates not passed** → uninformative about editability. Report the failure to learn as the
  result and move up the ladder; do **not** read an editability conclusion out of a model that
  never learned the game.

## Bootstrap — what a cold session must load

Nothing but the thread. `notebooks/experiments/editability/ours_on_othello/`:

```bash
cd notebooks/experiments/editability/ours_on_othello
python corpus.py 20000000            # ~3 h on 32 cores, 1.2 GB — CPU only, overlap it with training
python train.py --rung M --window 16 # ~18 min on one 5090
jupyter nbconvert --to notebook --execute --inplace ours_on_othello.ipynb
```

Model, corpora, gates, probes and editors are all defined in that directory; the registry
[`OURS_ON_OTHELLO_RUNS.md`](../../notebooks/experiments/editability/ours_on_othello/OURS_ON_OTHELLO_RUNS.md)
carries every configuration and every cited number. Nothing depends on live state from another
notebook.

## What is held fixed, and what moves

**Fixed — `runs/transformers/W16`'s architecture and recipe, verbatim:** `d_model` 256, 4 layers,
4 heads, `mlp_ratio` 4, pre-norm blocks, RoPE, banded-causal attention; AdamW lr 1e-3, weight decay
1e-4, batch 256, 5% warmup + cosine, grad-clip 1.0, `val_fraction` 0.1, best-val checkpointing.

**Moved — exactly three things:** `Linear(128, 256)+ReLU` → `nn.Embedding(61, 256)`;
`Linear(256, 128)` → `Linear(256, 61)`; MSE → cross-entropy. Sevan's call, 2026-08-21: replace the
**whole** encoder including the ReLU, mirroring the "no ReLU after the input projection" decision
already pinned for run A. Fresh init — no discworld weights are reused, because only the blocks
could transfer and whether they do is a different question.

**The band width is not a limitation.** `state_span = n_layers·(window−1)+1` = 61 at `window` 16,
so a full 60-move game is inside the receptive field. What the band costs is *directness* — their
model reaches any earlier move in one hop at all 8 layers, ours routes through up to 4. `window` 40
(span 157) runs alongside to remove that difference at **zero** extra compute.

## The control that makes it a fair test

Sevan, 2026-08-21: *"I don't really like how we are comparing very different data sizes … I don't
really know if success came from scale."*

Training at Li et al.'s 20M games is **222× the unique sequences** and **333× the unique tokens**
`W16` ever saw. So data scale is a **controlled axis**: every rung runs the **same 95,100 optimiser
steps** at batch 256 with the identical schedule, and only the pool size changes — 90k / 1M / 5M /
20M. Anything that moves across that row is diversity, not compute. Arm `F` (8 passes over 20M) is
the single exception, is flagged as such, and answers only "can this architecture do Othello at
all".

## Metrics and thresholds

**Gates — all held out, at `best_model.pt`, on games from a disjoint index range.** Arm `M` sees
90k games 300 times, so training loss is uninformative; the question is generalisation, exactly as
`W16` overfits discworld's training loss and is still a good predictor OOD.

| gate | threshold | note |
|---|---|---|
| legal-move mass | **≥ 0.95** for a "learned the game" reading | Li et al. report 0.9998 at 8× the parameters |
| excess CE over Bayes | report, no threshold | ⚠ the generator draws uniformly from the legal set, so a *perfect* model scores `mean(log\|legal\|)`. Raw CE is uninterpretable |
| probe error | must beat **random init at the same residual point** | absolute error rates are not comparable to Li et al.'s 1.7% / 20.4% at this scale |

**Editability** — `../../notebooks/experiments/editability/METRICS_AND_EDITORS.md` §6: Li error vs
post-flip, Li error vs pre-flip (the guard), Edit Index union and symdiff, legal mass, ‖Δx‖/‖x‖.
Four editors (Li gradient steering, Nanda addition, Nanda target−current, our pseudoinverse), each
written at **every** residual point and at **each single point**, because 2026-08-21 showed those
differ by 28× on their model.

⛔ **The Edit Index null must be recomputed per model, never borrowed.** On Li et al.'s checkpoint
the null is −0.829 *because that model predicts the unedited world well*; a weaker predictor starts
nearer 0, so an edit that does nothing can look better than one that works. **Report the gain over
each model's own null.** The random-init arm is in every table for exactly this reason.

## Mandated controls

1. **Random init**, identical architecture — Li et al.'s own `--random` arm — through the *whole*
   pipeline: gates, the full probe grid, and every editor.
2. **Seed disjointness**, asserted by hashing token rows, between training / test / probe-harvest
   corpora and the 1001-case benchmark.
3. **The `MLP ≥ linear` tripwire** on every probe cell.
4. **Two windows** (16 and 40), so a failure at 16 cannot be confused with a failure of the
   architecture.

## Known ways this could be uninformative

- **The model does not learn Othello at 90k games.** Likely, and not a bug — it is the parallel to
  discworld and moves the interesting question to the ladder.
- **The probe cache serving one model's probes to another.** Already happened once on 2026-08-21
  and is now prevented by keying the cache on a hash of the weights; `evaluate.attach` documents it.
- **Reading the Edit Index level instead of the gain.** See the ⛔ above.

## Expected artifacts

- notebook `ours_on_othello/ours_on_othello.ipynb`, registry `OURS_ON_OTHELLO_RUNS.md`
- `runs/ours_on_othello/` — `corpus/`, `<rung>_w<window>/`, `probe_cache/<fingerprint>/`,
  `figures/`, `results.json`
- scratch note `research/scratch/2026-08-21-ours-on-othello.md`
- `findings/editability.md` entry
