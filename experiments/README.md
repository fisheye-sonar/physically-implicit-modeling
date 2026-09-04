# experiments/ — the quarantined workspace

Every experiment is one folder here, and everything it produces stays inside it:

    experiments/<name>/
      README.md     what the question was, status, where the finding lives
      scripts/      the experiment's Python (pilots, builders, smokes)     [tracked]
      drivers/      its shell orchestration (chains, queues)               [tracked]
      scores/       its result JSON — small, the numbers a finding quotes  [tracked]
      data/  probes/  outputs/   bulk artefacts                            [gitignored]

Its **run logs** go to `logs/<name>/` — `logs/` holds logs and nothing else. `runs/` holds
trained runs and nothing else (`runs/_baselines/` and `runs/training_curve/` are scored by
the canonical `master_eval` and consumed by the tables, which is why they live there).

The point is quarantine: an experiment may be as messy as fast work needs, and the mess
cannot leak into `runs/`, `logs/`, `datasets/`, `outputs/` or the canonical `pim/` core. The
moment a number from here is quoted anywhere durable, the computation moves into `pim/` and
the script becomes a caller (CLAUDE.md §7). Every fitted probe is persisted — never refit
to look at one.

Generic queue/launch infrastructure that belongs to no experiment stays in
`scripts/drivers/` (`queue_after.sh`). Launch pattern for anything heavy:
`systemd-run --user --unit=<name> -p MemoryMax=…G --collect bash experiments/<name>/drivers/<x>.sh`
— one heavy job at a time, always with a watcher.

History: created 2026-09-02 when `logs/` had become the dumping ground `runs/` used to be;
the moves are ledgered in `logs/MOVES.md`.
