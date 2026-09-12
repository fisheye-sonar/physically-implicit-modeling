#!/usr/bin/env bash
# ── fit ONE extra probe target on ONE run (model probes + both floors), then score ───────
#   usage: probe_target_fit.sh <runs/<topic>/<run>> <target> [<unit/log name>]
#   The scorer never fits an extra target's probes (require_cached); this is the one place
#   they get fitted: the run's LIN + MLP (scripts/fit_probes.py, the target's recipe), the
#   random-init floor and the right-aligned observation floor on the run's instance, then
#   scripts/drivers/score_pending.sh (master_eval → build_full_table → headline).
# Under a unit:
#   systemd-run --user --unit=<name> -p MemoryMax=45G --collect --working-directory=$PWD \
#     /usr/bin/bash -c 'bash scripts/drivers/probe_target_fit.sh <run> <target> <name> > logs/<name>/unit.log 2>&1'
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$PWD
export PYTHONPATH=$ROOT
PY=$ROOT/.pim/bin/python
RUN=$1; TARGET=$2; NAME=${3:-probe_target_fit}
LOGS=$ROOT/logs/$NAME
NT=https://ntfy.sh/swirling-tornado-ai691k
mkdir -p "$LOGS"

ping()  { curl -sS --max-time 20 -H "Title: $1" -H "Tags: ${3:-information_source}" -d "$2" "$NT" > /dev/null 2>&1 || true; }
stage() { echo "=== [$(date '+%F %T')] STAGE $* ===" | tee -a "$LOGS/driver.log"; }
fail()  { ping "PIM $NAME FAILED: $1" "$2" warning; echo "FAILED: $1" >> "$LOGS/driver.log"; exit 1; }
fit()   { local tag=$1; shift
          "$PY" -u scripts/fit_probes.py --run "$RUN" --target "$TARGET" "$@" > "$LOGS/fit_$tag.log" 2>&1 \
            || fail "fit $tag" "$(tail -15 "$LOGS/fit_$tag.log")"; }

ping "PIM $NAME: started" "$TARGET on $RUN — model probes, random-init floor, observation floor, then scoring"
stage "fit model probes ($RUN · $TARGET)"
fit model
stage "fit random-init floor"
fit random_init --random-init
stage "fit observation floor"
fit observation --observation
stage "score"
bash scripts/drivers/score_pending.sh "$NAME" || fail "score_pending" "$(tail -5 "$LOGS/driver.log")"
