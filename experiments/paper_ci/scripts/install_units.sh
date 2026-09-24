#!/usr/bin/env bash
# ── install_units.sh — the systemd user units of the queue on THIS host ─────────────────────
#   bash experiments/paper_ci/scripts/install_units.sh lab      # dispatcher timer + watchdog timer + dashboard server
#   bash experiments/paper_ci/scripts/install_units.sh remote   # watchdog timer only
# Idempotent. Units live in ~/.config/systemd/user/; the repo path is substituted at install
# time so the same files serve both hosts. Requires linger (loginctl enable-linger) so the
# timers run without a login session — both hosts have it.
set -eu
HOST=${1:?lab|remote}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$ROOT/.pim/bin/python
U=$HOME/.config/systemd/user
mkdir -p "$U"

unit() {  # name, description, exec, [extra lines]
  cat > "$U/$1.service" <<EOF
[Unit]
Description=$2

[Service]
Type=oneshot
WorkingDirectory=$ROOT
Environment=PYTHONPATH=$ROOT
Environment=PIM_CI_HOST=$HOST
ExecStart=$3
TimeoutStartSec=${4:-900}
EOF
}
timer() {  # name, interval
  cat > "$U/$1.timer" <<EOF
[Unit]
Description=$1 every $2

[Timer]
OnBootSec=2min
OnUnitActiveSec=$2
AccuracySec=15s
Persistent=true

[Install]
WantedBy=timers.target
EOF
}

unit pimci-watchdog "PIM CI watchdog ($HOST)" "$PY $ROOT/experiments/paper_ci/scripts/watchdog.py --host $HOST" 300
timer pimci-watchdog 5min
if [ "$HOST" = "lab" ]; then
  unit pimci-dispatch "PIM CI dispatcher tick" "$PY $ROOT/experiments/paper_ci/scripts/dispatch.py" 1500
  timer pimci-dispatch 2min
  cat > "$U/pimci-dashboard.service" <<EOF
[Unit]
Description=PIM CI dashboard (static files on 127.0.0.1:8765, published by tailscale serve)

[Service]
WorkingDirectory=$ROOT/experiments/paper_ci/dashboard
ExecStart=$PY -u $ROOT/experiments/paper_ci/scripts/serve.py 8765
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
fi
systemctl --user daemon-reload
systemctl --user enable --now pimci-watchdog.timer
if [ "$HOST" = "lab" ]; then
  systemctl --user enable --now pimci-dispatch.timer
  systemctl --user enable --now pimci-dashboard.service
fi
systemctl --user list-timers --no-legend pimci-* || true
systemctl --user is-active pimci-dashboard.service 2>/dev/null || true
