#!/usr/bin/env bash
# ── oth-adjacent-flip seed variance: two re-seeded replicates + probe reseeding (2026-09-16) ──
#
# Sevan: variance estimates for `adjacent_flip_ablation/L-oth-adjacent-flip-20m` — the run's
# contrasts (IM +0.66/0.51 symdiff, ND +0.58, PI +0.40) are quoted in the paper with no spread.
# Two sources of variance, the seed-variance convention (`experiments/seed_variance/README.md`):
#   RUN seeds   — re-train at 390k steps with --seed 1 / --seed 2 (`scripts/drivers/replicate.sh`),
#                 plus the parent's own step-421,875 checkpoint as the seed-0 member of the set
#                 (a matched budget, inside the tables' ±10% pooling window). Val loss at 390k is
#                 within 0.1% of the 780k best on this run, so the budget costs nothing.
#                 Replicates are NEVER table rows: `build_full_tables` pools them into the parent
#                 row's ± column. The training corpus is NOT regenerated.
#   PROBE seeds — 10 seeds of the linear mine/theirs grid AND of the inverse map g (the seed drives
#                 init and the 80/20 split by game, so IM-NN's retrieval bank is reseeded with it),
#                 PI / ND / IM / IM-NN re-swept per seed → `runs/<run>/variance.json` (Table 4b).
#
# ORDER (Sevan): the first reseed is fully trained, scored AND probed before the second starts.
# Every stage is idempotent and the chain is safe to stop: `systemctl --user stop oth_adjflip_seeds`,
# then relaunch the same unit command — finished stages skip and an interrupted training resumes
# from its own ckpt/latest.pt (replicate.sh passes --resume when one exists).
#
#   systemd-run --user --unit=oth_adjflip_seeds -p MemoryMax=45G --collect --working-directory=$PWD \
#       /usr/bin/bash -c 'bash scripts/drivers/oth_adjflip_seeds.sh > logs/oth_adjflip_seeds/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
PY=$ROOT/.pim/bin/python
export PYTHONPATH=$ROOT
export PIM_SKIP_TOPICS=training_curve      # ⛔ paused by Sevan 2026-09-15 — master_eval must not enter them
NT=https://ntfy.sh/swirling-tornado-ai691k
TOPIC=adjacent_flip_ablation
PARENT=$TOPIC/L-oth-adjacent-flip-20m
STEPS=390000
SEEDS=10                                   # probe seeds (linear grid AND inverse map)
LOGS=$ROOT/logs/oth_adjflip_seeds
mkdir -p "$LOGS"
echo $$ > "$LOGS/driver.pid"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM oth-adjflip-seeds FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

# probe seeds on ONE run, skipped when that run already carries the full set
probe_seeds() {
  local run=$1 tag=$2
  if "$PY" - "$run" "$SEEDS" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path("runs") / sys.argv[1] / "variance.json"
n = int(sys.argv[2])
if not p.exists():
    sys.exit(1)
v = json.loads(p.read_text()).get("probe_seeds", {})
ok = (v.get("mine", {}).get("summary", {}).get("n_seeds", 0) >= n
      and v.get("inverse_map", {}).get("summary", {}).get("n_seeds", 0) >= n)
sys.exit(0 if ok else 1)
PYEOF
  then
    echo "  probe seeds already complete on $run — skipping" | tee -a "$LOGS/driver.log"
    return
  fi
  "$PY" -u experiments/seed_variance/scripts/probe_seeds_othello.py --run "$run" --seeds "$SEEDS" \
      > "$LOGS/${tag}.log" 2>&1 || fail "probe seeds $run" "$(tail -15 "$LOGS/${tag}.log")"
  ping "PIM oth-adjflip-seeds: probe seeds done ($run)" "$(tail -3 "$LOGS/${tag}.log")"
}

ping "PIM oth-adjflip-seeds: chain started" \
"parent probe seeds (~30 min) -> seed 1 train ${STEPS} (~10.5 h) + score + probe seeds (~1 h) -> seed 2 (same). ETA ~14:00 tomorrow. Safe to stop and relaunch."

# ── A — probe seeds on the parent (no dependency on the replicates; the earliest number) ──
stage "A probe seeds on the parent ($SEEDS seeds: linear grid + inverse map)"
probe_seeds "$PARENT" "a_probeseeds_parent"

# ── B — replicate seed 1: train, lay out the seed-0 checkpoint member, score, tables ──
stage "B replicate seed 1 (train $STEPS + seed-0 ckpt member + score)"
bash scripts/drivers/replicate.sh "$PARENT" 1 "$STEPS" || fail "B replicate seed 1" "$(tail -15 "$LOGS/../rep_L-oth-adjacent-flip-20m_s1/driver.log" 2>/dev/null)"
CKPT_RUN=$(ls -d "$ROOT/runs/$TOPIC/L-oth-adjacent-flip-20m__seed0_s"* 2>/dev/null | head -n 1 | xargs -n1 basename)
[ -n "$CKPT_RUN" ] || fail "B ckpt member" "replicate.sh left no __seed0_s* run"

# ── C — probe seeds on the new replicate AND on the seed-0 checkpoint member ──
stage "C probe seeds on seed 1 and the seed-0 member ($CKPT_RUN)"
probe_seeds "$TOPIC/L-oth-adjacent-flip-20m__seed1" "c_probeseeds_seed1"
probe_seeds "$TOPIC/$CKPT_RUN" "c_probeseeds_seed0ckpt"
ping "PIM oth-adjflip-seeds: SEED 1 COMPLETE" \
"$("$PY" - "$TOPIC" <<'PYEOF'
import json, glob
from pathlib import Path
for p in sorted(glob.glob(f"runs/{__import__('sys').argv[1]}/L-oth-adjacent-flip-20m*/scores.json")):
    s = json.load(open(p))
    b = {e: v for e, v in s["best"].items() if v}
    print(Path(p).parent.name, "val %.4f" % s["val_loss"],
          " ".join(f"{e} {v.get('edit_index_symdiff', v.get('edit_index')):+.3f}" for e, v in b.items()))
PYEOF
)
Starting seed 2 (~10.5 h)." white_check_mark

# ── D — replicate seed 2 ──
stage "D replicate seed 2 (train $STEPS + score)"
bash scripts/drivers/replicate.sh "$PARENT" 2 "$STEPS" || fail "D replicate seed 2" "$(tail -15 "$LOGS/../rep_L-oth-adjacent-flip-20m_s2/driver.log" 2>/dev/null)"

# ── E — probe seeds on seed 2 ──
stage "E probe seeds on seed 2"
probe_seeds "$TOPIC/L-oth-adjacent-flip-20m__seed2" "e_probeseeds_seed2"

ping "PIM oth-adjflip-seeds: ALL DONE" \
"3-member replicate set (seeds 0-ckpt / 1 / 2) + $SEEDS probe seeds each; tables carry the parent's ± column." white_check_mark
stage "chain complete"
