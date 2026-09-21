# queue drafts — NOT installed (Sevan's call, 2026-09-20)

Catch-up jobs for the categorical inverse map, in the schema of `experiments/paper_ci/queue/`. To install after
the branch is merged and `clear_continuous_im.py --apply` has run: copy the wanted files into
`experiments/paper_ci/queue/` on the lab and run `plan.py --show` (a job file that exists is never rewritten).
Priorities 0-4 put them ahead of every queued replicate, in this order; a `gpu`-lane job still waits for the job
RUNNING on that host to end. (2026-09-20 evening: all three dw-128ray members were scored with the OLD arm by 13:25, so
128-ray has its own catch-up, `catim_128ray` — 17 in-scope runs in all: 4 parents, 12 members, the token model.)

`score_pending.sh` holds no lock: two `master_eval` executions on one host would both write the notebook in place.
In the `gpu` lane that cannot happen (one job per host). If a catch-up is run as a side unit instead (sharing the
GPU with a training stage, as the previews did, ~20% slowdown of the training job), it must end before that host's
replicate reaches its scoring stage.

Order matters: the driver adds the arm only to a block with NO IM arm (`driver.missing_inverse`), so a run must be
cleared (`clear_continuous_im.py --apply`) before its catch-up job, or the job finds nothing to do. `PIM_ONLY_RUNS`
matches run names exactly (`pim/scoring/runs.py`). `final_tables`'s deps were fixed at plan time and do not name these
jobs. ⚠ Do NOT hand-edit queue files to install these (the queue operator's note, PROGRESS 2026-09-20 14:20): `plan.py`
REGENERATES every still-queued job it builds, so a hand-added id or dependency is dropped at its next run. Register the
catch-up jobs in `plan.py` (as `appendix_prediction` and the controls are) — these files are the content to register.
