# scripts/drivers — generic run-queue infrastructure

Only orchestration that belongs to **no particular experiment** lives here:

- `queue_after.sh <unit> <command…>` — wait for a systemd user unit to finish (or fail),
  then run the command. One heavy job at a time.

Every experiment's own drivers (training chains, scoring chains) live with the experiment
in `experiments/<name>/drivers/` — see `experiments/README.md`. Moved there 2026-09-02.
