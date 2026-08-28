# 2026-08-20 — Our probe and our editor, on Li et al.'s Othello-GPT

Thread `notebooks/experiments/editability/othello_transfer/`, notebook `probe_transfer.ipynb`.
Outputs in `runs/othello_transfer/` (5 figures, `results.json`, `probe_cache/`).
20,000 synthetic games → 1,179,401 (activation, board) rows; 108 probes; 72 intervention arms;
all 1001 benchmark cases per arm. ~72 min end to end. No world models trained.

## Why

`othello_gpt/` (2026-08-18) ran **their method on our model** and found the probing half replicates
while the intervention half does not. Three explanations survived: the world, the read-out, or
**our editor implementation**. The third could not be tested from inside the thread, because every
editability number in the repo comes from that same code. This runs the mirror: **our probe and
our editor, unmodified, on their model, against their own benchmark.**

## Headline — it reproduces, and then some

| | Li et al. | ours, our code, their model |
|---|---|---|
| null intervention | 2.68 | **2.723** |
| best intervention | 0.12 (at `L_s`=4) | **0.016** |
| error reduction | 22x | **170x** |
| nonlinear probe, best layer | 1.7% error | **0.57%** |
| linear probe, best layer | 20.4% error | 23.90% (absolute colour) |

**Our editor implementation is cleared.** The discworld editability negative is not a bug in
`othello_probe.py`'s probe fitting, its edit objective, its descent, or its multi-layer schedule —
that exact code produces a 170x error reduction on the model the published result was published on.

## In Edit Index terms — the axis that compares the two worlds

Same formula, same ±1 scale. Reference world = uniform over legal moves, which is *exact* here:
their generator draws moves uniformly from the legal set, so it is the Bayes-optimal predictor.

| | unsteered | best arm | gain | crosses zero |
|---|---|---|---|---|
| **Othello** — their model, our code | **−0.829** | **+0.656** | **+1.49** | **yes** |
| **discworld** — our model, our code (`othello_gpt/`, 2026-08-19) | −0.684 | −0.194 | +0.49 | **no** |

81% of the available headroom captured against 29%. The qualitative fact is the sign: on Othello
the output stops being the unedited world and becomes the edited one; on discworld it never leaves
the unedited world.

On the narrower symmetric-difference support the unsteered floor is −0.943 and the sweep reaches
**+0.868** — on the squares whose legality actually changed, the prediction is most of the way to
the edited world.

## Three things that came out of it

**1. Nanda's linear finding reproduces exactly.** Linear probe, best layer: **23.90%** error in
absolute colour, **0.72%** in mine/theirs. Li's "linear probes fail" (20.4%) is a coordinate-frame
artifact — the board is linearly decodable once the alternating sign is accounted for. Their
*nonlinear* probe was reading a representation that was linear all along.

**2. The frame/sequence split barely matters here** — 0.57% vs 0.66% at the best layer, ~0.1 pp.
On discworld the same convention change inflated velocity R² by +0.34 (`GOTCHAS.md`, 2026-08-14).
Board state changes enough per move that a frame split leaks little, so **their published number is
not inflated by their split convention**. Cell [15] also shows the probe's split convention does not
change the intervention outcome.

**3. The probe constraint does not pin down the write, on their model either.** `hit_target` — the
probe reads the requested board, their own success criterion — is **1.000 at every alpha across a
50x range**, and the edit objective reaches ≥99% of its best reduction everywhere too. Over that
same range the outcome moves by a factor of **83** (100 steps) / **107** (1000 steps). So no
write-side criterion can select a step size: what decides whether the dynamics honour the write is
its **magnitude**, not its satisfaction of the probe. This is the 2026-08-18 discworld observation
("the optimiser decides which probe-satisfying write you land on") reproduced on their model, which
makes it a property of probe-derived writes rather than of our world.

## Other numbers worth keeping

- **Best arms** (all `MLP 512 hidden`, α = 0.02, `L_s` = 2): by their metric, 1000 steps →
  Li 0.016, pre-flip 2.214, Edit Index +0.646, legal mass 0.998. By Edit Index, 100 steps →
  Li 0.018, Edit Index **+0.656**.
- **The guard works.** Best arm: legal mass 0.858 → **0.998**, pre-flip error 0.002 → 2.214. It
  left the old world rather than degrading. At α = 0.1 the arms move up-and-right in Fig 4 —
  degradation, visible only because both worlds are scored.
- **Sharp transition at `L_s` = 5** (Fig 3): writing only at the last three or four residual points
  fails, Edit Index collapsing from ~+0.6 to −0.5. Matches the structural prediction that a write
  at point ℓ changes block inputs only for layers > ℓ.
- **The two metrics disagree about the best step size.** Li error is minimised at α = 0.05; the
  Edit Index at α = 0.02. "Closest to the new world" is not "cleanly *is* the new world", which is
  what the Edit Index exists to separate.
- **Benchmark identified as the paper's NATURAL subset** — our null baseline 2.723 against their
  published 2.68 natural / 2.59 unnatural. 1001 cases, all integrity checks clean; the flip changes
  2.1 of 64 squares on average.

## Caveats, stated

- **MLP ≥ linear tripwire fired 8 times**, all on mine/theirs where both probes sit under 3.2% and
  the largest gap is 0.19 pp. Reading: both at the ceiling of an easy target, not an undertrained
  MLP — on absolute colour the ordering holds by 23 pp. Not dismissed, but not treated as the bug
  the tripwire is calibrated for.
- **Deviations from Li et al., all deliberate**: synthetic rather than championship games (their
  script hardcodes championship even for this checkpoint; that data is behind a dead link);
  `model.eval()` (their harvest runs with dropout live at p=0.1); input standardisation inside the
  probe (ours, and why their `lr = 1e-3` does not transfer); 20k games rather than ~130k.
- **This does not discriminate** between the two remaining explanations for the discworld negative —
  the world (discrete board consumed directly by the legal-move computation vs continuous positions
  reaching the output only through a renderer) and the read-out (their probe predicts a quantity the
  computation demonstrably consumes; ours one merely correlated with it). Both survive.

## Open

- The obvious next test is the read-out explanation: a discworld probe trained on a quantity the
  decoder provably consumes. Flagged as a candidate direction on 2026-08-18 and not yet briefed.
- Everything here is **step 0 only**, matching their benchmark. Persistence — where discworld's
  edits actually die (+0.146 → +0.010 by step 14) — is untested here and needs its own design
  (what the counterfactual board does while the model plays on, whether moves are sampled or
  argmaxed). Deliberately deferred, per Sevan.
- Their checkpoint came from a third-party mirror after all four of the paper's Google Drive links
  were found dead; verified functionally identical to the authors' own TransformerLens conversion
  (2.1e-6). Championship data is ~93% reconstructible from the public WTHOR archive (136,055 games,
  parsed and validated) if we ever need it.
