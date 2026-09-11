# blink_ablation — is position editable once it MUST be carried? (2026-09-07)

**Question.** In every discworld instance so far the current frame shows every object, so
the model never has to remember a position; the flip ablation showed that a variable the
dynamics do not USE is decodable but not editable. dw-blink removes an object from the
observation for a run of frames (physics unchanged, a 0.5 marker on its edge ray the
frame before and the last hidden frame — `pim/environments/discworld/blink.py`). Predicting
the reappearance frame needs a remembered, advanced position, so position is a guaranteed
member of the causal state. Is it editable there?

**Status (2026-09-08): DONE, negative.** Position is carried through blackouts (MLP 0.97–0.99 on hidden frames) and not editable on any subset; write-up `research/findings/blink-ablation.md`.

**Instance / run.** `datasets/discworld/dw-blink/instance.json`; run
`runs/blink_ablation/L-dw-blink-20m` (Transformer-L, 20M, 780k steps, the matched recipe);
driver `scripts/drivers/dw_blink.sh`; logs `logs/blink_ablation/dw_blink/`.

**Layout.**
- `pilot/` — the 3k/6k-case pilot split (seeds 200e9, throwaway) and `stats.json`, the gates
  run before the 20M corpus (`scripts/pilot_stats.py`): warm-up rule, never-both-hidden,
  marker placement, hidden-object leaks, subset sizes, zones NaN exactly on hidden cases.
- `scripts/subset_editability.py` — after training: the canonical editors on three case
  subsets of the 20k edits (reappearance at frame 20 with staleness k, mid-blackout scored
  by step at reappearance, visible control), both bases; plus the hidden-frame decodability
  split (probe skill on hidden vs visible frames, against the observation floors).
- `scores/` — its outputs.

**2026-09-10 — `scripts/hidden_frame_floors.py`:** the random-init and observation probes (cached) evaluated on the SAME hidden frames as the trained model's decodability table; trained MLP 0.96–0.98 at ten frames since seen vs observation 0.46–0.78 and random-init 0.58–0.74 — position is computed and carried, and still not editable (regression target). `scores/hidden_frame_floors.json`. `research/findings/blink-ablation.md`.
