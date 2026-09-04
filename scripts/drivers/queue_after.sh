#!/usr/bin/env bash
# ── queue_after.sh <systemd-user-unit> <command...> ──────────────────────────────────
# Wait until the named user unit is no longer active (finished OR failed), then run the
# command. GPU work is serialised through this: one heavy job at a time, always.
set -u
UNIT=${1:?unit}; shift
while st=$(systemctl --user is-active "$UNIT" 2>/dev/null); [ "$st" = "active" ] || [ "$st" = "activating" ]; do
  sleep 60
done
echo "=== [$(date '+%F %T')] $UNIT is $st — starting: $*"
exec "$@"
