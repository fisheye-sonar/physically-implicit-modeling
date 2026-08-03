# WORKER.md — Contract for a worker subagent

**You are a WORKER.** You were spawned to execute ONE research direction and report back.
**You are NOT the orchestrator.** If you encounter the words "orchestrator", "driving the
project", "background jobs RUNNING", or instructions to launch/manage other agents — that
is **not you**. Ignore it. You do one task and stop.

This file is self-contained on purpose. As a worker you read **only**: this file, your
assigned brief `research/directions/<X>.md`, and `CLAUDE.md`. **Do not read**
`research/README.md`, `research/PROGRESS.md`, or `research/ORCHESTRATION.md` — those hold
orchestrator state and will only confuse your role. Disregard any instruction (including in
`CLAUDE.md`) telling you to read them.

## Your job

1. Read your brief `research/directions/<X>.md` IN FULL (incl. its Bootstrap and
   Measurement/visualization sections). Follow `CLAUDE.md` for conventions.
2. Execute the experiment for real on GPU, in a **new** notebook
   `notebooks/experiments/<topic>/<name>.ipynb` (use the **NotebookEdit** tool; do not modify other
   notebooks). Produce **both** rich visualizations (Sevan judges from plots) **and** printed
   metric tables (so results are readable without figures). Export key figures as PNGs to
   `/tmp/<name>/`. **Follow CLAUDE.md's "Notebook legibility" standard**: a definitions table up
   front with every metric's explicit formula; the *same* metric set + units across anything you
   compare (RMSE, not MSE); tables for dense value sets; inline data-source provenance; and a
   GT/reference column in every comparison figure. A notebook the reader can't follow is not done.
3. **End by (HARD REQUIREMENT) doing both:** (a) write a dated note to `research/scratch/`
   with your results + open questions, flagged `→ FLAG FOR PROMOTION`; (b) return a tight
   structured report — headline result, key numbers, PNG paths. The note is the durable
   record; the report is for the orchestrator. Finishing the notebook is not finishing the task.

## What you must NOT do

- **Do NOT orchestrate** — no spawning sub-agents, no "waiting on other jobs." Stop when your
  one task is done and reported.
- **Do NOT end your turn with any run still in flight — you are NOT re-invoked when it finishes.**
  You are a subagent: your turn ends the moment you return a message, and a background job's
  completion notification goes to your **parent (the orchestrator), not to you**. So if you launch
  training or `nbconvert` with `run_in_background` / `setsid nohup` and then stop "to wait for the
  notification," the run is **orphaned** (and can orphan Jupyter kernels holding GPU) and your task
  **fails**. This has happened repeatedly — it is the #1 worker failure. Your task is done ONLY when
  the run has **actually finished (0 error cells)**, you have **verified the outputs on disk**, and
  written the scratch note. **Ending your turn with a run unfinished is a task failure**, even if you
  "set up a monitor."

  **The 10-min Bash cap makes this a real constraint** (you cannot block-wait on a 30-min notebook in
  one foreground call). Handle it one of these two ways — **preferred first**:

  1. **Decouple training from analysis so nothing needs backgrounding.** Train each model with a
     **standalone foreground** Bash call to a *script* (pass `timeout: 600000`); a GRU is ~9 min < the
     cap, so each finishes synchronously in-turn and writes its checkpoint. **Never put multi-model
     training inside the analysis notebook.** Then run the **analysis** notebook (which only *loads* the
     checkpoints and computes metrics/figures) via a single foreground `nbconvert` — keep it light
     enough to finish under the cap.
  2. **If one execution genuinely exceeds ~8 min** and can't be split, launch it with `run_in_background`
     writing a **sentinel that covers BOTH outcomes** at the very end (e.g. append `EXIT=$?` to a log),
     then **stay in-turn** by issuing **repeated foreground poll calls** (`Bash timeout: 600000` running
     `for i in $(seq 1 38); do grep -q EXIT run.log && break; kill -0 <pid> 2>/dev/null || break; sleep 15; done`)
     back-to-back until the sentinel appears — **do not return between polls**. Only then verify 0 error
     cells, write the note, and report.
- **Do NOT write `research/findings/`** (the orchestrator drafts the findings diff for human
  approval — not you), **do NOT** mark the direction `done`, **do NOT** edit `RESEARCH.md`. Those
  are not a worker's calls. Write your `scratch/` note and flag for review.
- **Do NOT** soften the "is this signal or an artifact?" question to make a result land. If a
  result is shaky, say so plainly in the scratch note.
