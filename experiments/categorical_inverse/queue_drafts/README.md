# queue drafts — NOT installed (Sevan's call, 2026-09-20)

Catch-up jobs for the categorical inverse map, in the schema of `experiments/paper_ci/queue/`. To install after
the branch is merged and `clear_continuous_im.py --apply` has run: copy the wanted files into
`experiments/paper_ci/queue/` on the lab and run `plan.py --show` (a job file that exists is never rewritten).
Priorities 0-4 put them ahead of every queued replicate, in this order; a `gpu`-lane job still waits for the job
RUNNING on that host to end. dw-128ray has no members yet: they are scored fresh after the merge and get the arm
with no flag.

`score_pending.sh` holds no lock: two `master_eval` executions on one host would both write the notebook in place.
In the `gpu` lane that cannot happen (one job per host). If a catch-up is run as a side unit instead (sharing the
GPU with a training stage, as the previews did, ~20% slowdown of the training job), it must end before that host's
replicate reaches its scoring stage.

Order matters: the driver adds the arm only to a block with NO IM arm (`driver.missing_inverse`), so a run must be
cleared (`clear_continuous_im.py --apply`) before its catch-up job, or the job finds nothing to do. `PIM_ONLY_RUNS`
matches run names exactly (`pim/scoring/runs.py`). `final_tables`'s deps were fixed at plan time and do not name these
jobs; at priority 0-4 they finish days before it, but add their ids to `queue/final_tables.json` when installing
(it is still `queued`, so editing it is safe).
