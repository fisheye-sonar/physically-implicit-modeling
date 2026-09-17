#!/usr/bin/env bash
# ── replicate.sh — ONE seed replicate of a canonical run, end to end (2026-09-14) ─────────
#   usage: replicate.sh <topic/parent> <seed> <steps> [<categorical target> ...]
#   e.g.   replicate.sh ray_ablation/L-dw-8ray-20m 1 390000 appearance-fac appearance
#
#   A  train runs/<topic>/<parent>__seed<seed> at <steps> with --replicate-of (skipped if done)
#   B  lay out the parent's saved checkpoint NEAREST <steps> as <parent>__seed0_s<step>
#      (skipped if one exists) — the seed-0 member of the replicate set at a matched budget
#   C  fit the parent's categorical extra targets on the new replicate (model probes; the
#      instance's floors are cache hits) — the regression probes are fitted by the scorer
#   D  score_pending.sh: master_eval adds the replicate's blocks (it inherits the parent's
#      extra targets), the tables fold it into the parent's ± (matched budget)
# Discworld only for stage C (Othello's extra targets are fitted inline by the scorer).
# Under a unit:
#   systemd-run --user --unit=rep_<name> -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/replicate.sh <topic/parent> <seed> <steps> [targets] > logs/rep_<name>/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
PARENT=$1; SEED=$2; STEPS=$3; shift 3; TARGETS=("$@")
TOPIC=${PARENT%/*}; PNAME=${PARENT#*/}
RUN=${PNAME}__seed${SEED}
NAME=rep_${PNAME}_s${SEED}
LOGS=$ROOT/logs/$NAME
NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"
ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }

CFG=$ROOT/runs/$PARENT/config.json
[ -f "$CFG" ] || fail "parent" "no config.json at runs/$PARENT"
ENV=$("$PY" -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['data'].get('env','discworld') if 'env' in c['data'] else ('othello' if 'oth' in c['data'].get('instance','') else 'discworld'))" "$CFG")
INST=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['data']['instance'])" "$CFG")
ARCH=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['arch'])" "$CFG")
ARCH_FLAG=${ARCH%_tokens}; REPR_FLAG=(); [[ "$ARCH" == *_tokens ]] && REPR_FLAG=(--repr tokens)
ping "PIM $NAME: started" "replicate of $PARENT, seed $SEED, $STEPS steps; targets: ${TARGETS[*]:-none}"

stage "A train $RUN ($ENV / $INST / $ARCH, $STEPS steps, seed $SEED)"
if [ -f "$ROOT/runs/$TOPIC/$RUN/best_model.pt" ] && [ -f "$ROOT/runs/$TOPIC/$RUN/metrics.jsonl" ] \
   && tail -n 1 "$ROOT/runs/$TOPIC/$RUN/metrics.jsonl" | grep -q "\"step\": $STEPS"; then
  echo "  already trained — skipping" | tee -a "$LOGS/driver.log"
else
  # an interrupted replicate resumes from its own ckpt/latest.pt (train.py refuses the dir otherwise)
  RESUME=()
  [ -f "$ROOT/runs/$TOPIC/$RUN/ckpt/latest.pt" ] && { RESUME=(--resume); echo "  resuming from ckpt/latest.pt" | tee -a "$LOGS/driver.log"; }
  "$PY" scripts/train.py --env "$ENV" --arch "$ARCH_FLAG" "${REPR_FLAG[@]}" --instance "$INST" \
      --topic "$TOPIC" --run-name "$RUN" --steps "$STEPS" --seed "$SEED" --replicate-of "$PARENT" "${RESUME[@]}" \
      > "$LOGS/a_train.log" 2>&1 || fail "A train" "$(tail -20 "$LOGS/a_train.log")"
  ping "PIM $NAME: trained" "$(grep '^done' "$LOGS/a_train.log" | tail -1)"
fi

stage "B parent checkpoint nearest $STEPS as the seed-0 member"
"$PY" experiments/seed_variance/scripts/layout_checkpoint_replicate.py "$PARENT" "nearest:$STEPS" \
    > "$LOGS/b_layout.log" 2>&1 || fail "B layout" "$(tail -10 "$LOGS/b_layout.log")"
CKPT_RUN=$(ls -d "$ROOT/runs/$TOPIC/${PNAME}__seed0_s"* | head -n 1 | xargs -n1 basename)

if [ "$ENV" = "discworld" ] && [ ${#TARGETS[@]} -gt 0 ]; then
  stage "C categorical targets on $RUN and $CKPT_RUN: ${TARGETS[*]}"
  for T in "${TARGETS[@]}"; do
    for R in "$TOPIC/$RUN" "$TOPIC/$CKPT_RUN"; do
      "$PY" -u scripts/fit_probes.py --run "$R" --target "$T" > "$LOGS/c_fit_$(basename "$R")_$T.log" 2>&1 \
          || fail "C fit $T on $(basename "$R")" "$(tail -15 "$LOGS/c_fit_$(basename "$R")_$T.log")"
    done
    # the instance's floors for this target (cache hits when the parent already has them)
    "$PY" -u scripts/fit_probes.py --run "$PARENT" --target "$T" --random-init > "$LOGS/c_floor_rand_$T.log" 2>&1 || fail "C random-init floor $T" "$(tail -15 "$LOGS/c_floor_rand_$T.log")"
    "$PY" -u scripts/fit_probes.py --run "$PARENT" --target "$T" --observation > "$LOGS/c_floor_obs_$T.log" 2>&1 || fail "C observation floor $T" "$(tail -15 "$LOGS/c_floor_obs_$T.log")"
  done
fi

stage "D score + tables"
bash scripts/drivers/score_pending.sh "$NAME" || fail "D score_pending" "$(tail -5 "$LOGS/driver.log")"
ping "PIM $NAME: DONE" "$RUN and $CKPT_RUN scored; folded into $PNAME's ± in the tables." white_check_mark
stage "chain complete"
