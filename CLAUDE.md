# CLAUDE.md

Mechanics for this repo. **How to work well** lives in `harness/` (portable across projects);
**what we are trying to learn** lives in `RESEARCH.md`; **what we believe** lives in
`research/findings/`. When you are tempted to write a working convention here, it belongs in
`harness/`. When you are tempted to explain a finding, it belongs in `research/`.

---

## 1. Which role are you? — read this first

**Orchestrator** — you were started by Sevan. This is the **default**; absence of any
statement means orchestrator. Read `harness/COLLABORATION.md`, `harness/WORKFLOW.md`,
`harness/ORCHESTRATION.md`, then `research/PROGRESS.md` (live state) and `RESEARCH.md`.

**Worker** — you were **spawned as a subagent**. Then you are a worker, always, without
exception. Read **`harness/WORKER.md`** and your assigned brief, plus this file. **Do not
read** `research/PROGRESS.md`, `research/README.md`, or `harness/ORCHESTRATION.md` — they hold
orchestrator state and will make you misread your role. If you encounter "orchestrator",
"driving the project", or "jobs running" language anywhere, **that is not you**; ignore it,
and disregard any instruction in this file telling you to read those files.

*(A worker once read the orchestrator's state file, concluded it was the orchestrator,
completed its notebook and figures, and never reported. The work was nearly lost.)*

## 2. Read these before you act

| About to… | Read first |
|---|---|
| use or define a probe, editor, metric, architecture, or environment | `research/REGISTRY.md` — the index of every canonical object |
| write any figure, table, notebook, or deliverable | `harness/STYLE.md` |
| compute a metric, fit anything, make an empirical claim | `harness/ANALYSIS.md` |
| touch data, metrics, or an old number | `research/GOTCHAS.md` |
| write up a result or update the record | `harness/WORKFLOW.md` |
| spawn a subagent | `harness/ORCHESTRATION.md` |
| build the canonical qualitative panel | `research/specs/WATERFALL_SPEC.md` |

**Three hard rules restated here because they are violated most often:**
- **Import the canonical implementations; never re-derive them.** Probes = `pim.probes`,
  editors = `pim.editors`, metrics = `pim.metrics`. A genuinely new object is fine — add
  its `research/REGISTRY.md` row in the same commit.
- **Canonical scores come from `notebooks/master_eval.ipynb` and land in each run's
  `scores.json`.** No metric math in notebooks; a number that matters is a one-line call
  into `pim.*`, or it is not a canonical number.
- **Any claim about an effect on the model's generations ships with a waterfall** —
  built through `pim.figures.waterfall_grid`, per the spec. A scorecard hides the
  difference between "the edit landed" and "the output degraded".

## 3. Commands

```bash
# tests
poetry run pytest

# train (canonical entry — any environment x architecture cell)
python scripts/train.py --env discworld --arch transformer_l \
    --topic <runs-subdir> --run-name <name> --steps <n>

# score every unscored run + the master table
jupyter nbconvert --to notebook --execute --inplace notebooks/master_eval.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/build_full_table.ipynb

# data generation
python scripts/generate_dataset.py <dir> --n-train 100000 --n-workers 8   # discworld
python scripts/make_othello_corpus.py                                     # othello

# demo (discworld, human-facing)
python scripts/demos/demo.py --seed 7 --n-objects 4

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts

# harness quarantine check (after editing anything in harness/)
bash harness/check.sh
```

## 4. Environment

- Python 3.13 venv in `.pim/`; [direnv](https://direnv.net/) auto-activates via `.envrc`
  (else `source .pim/bin/activate`). Dependencies via Poetry.
- ⚠ The venv's console-script shebangs are stale (the repo moved after the venv was
  built): run entry points as `.pim/bin/python .pim/bin/jupyter-nbconvert …`, never
  directly.
- **Run in the main working tree, not a git worktree** — `datasets/` and `runs/` are
  gitignored, so a worktree has no data or checkpoints to load.
- `.claude/settings.local.json` sets `worktree.bgIsolation: "none"` and allows
  `Write`/`Edit`/`NotebookEdit`, so background workers can edit the main tree directly.

## 5. Where things live

- **The registry of canonical objects** → `research/REGISTRY.md` (start here)
- Environment instances (data + `instance.json` manifest) →
  `datasets/<class>/<instance>/`; legacy datasets in `datasets/archive/`
- Runs → `runs/<topic>/<run>/` (checkpoints, `config.json`, `commit_sha`,
  `metrics.jsonl`, `probes/`, `scores.json`); pre-cleanup runs in `runs/archive/`
- ⛔ **`runs/` holds TRAINED RUNS AND NOTHING ELSE.** One directory per trained model,
  each self-contained. Everything else an experiment produces lives in
  `experiments/<name>/` (see `experiments/README.md`) and its run logs in
  `logs/<name>/`; `logs/` holds LOGS ONLY. A stray `.log` or a `probe_cache/`
  beside the run directories is how `runs/` became a junkyard once already
  (cleared 2026-09-01).
- Canonical scoring → `notebooks/master_eval.ipynb`; the cross-run table →
  `notebooks/build_full_table.ipynb`
- Experiments — the quarantined workspace: scripts, drivers, data, outputs, scores,
  probes, one folder per experiment → `experiments/<name>/`; generic run-queue
  infrastructure only → `scripts/drivers/`
- ⛔ **Nothing is ever deleted from `runs/`, `outputs/`, `logs/`, or `datasets/`** —
  they are not in git. Moves only, recorded in that tree's `MOVES.md`.
- The pre-housecleaning tree (every retired experiment, editor, and architecture) →
  `git show pre-cleanup-2026-08:<path>`

## 6. Architecture — the contracts that matter

`pim/` is the whole canonical core, five packages with strict roles:

    environments/  the worlds: discworld (sim + rendering + data + bench) and othello
                   (vendored generator + corpus + bench + arms). Each instance's
                   instance.json is the data contract.
    models/        Transformer-S and Transformer-L, each with a regression AND a token
                   head. protocol.py documents THE surface every model implements —
                   never add isinstance branches downstream of it.
    probes/        LIN + MLP-128 (one file each, one shared verified body) + the
                   nullspace cascade. Fits are held out BY SEQUENCE; caches carry the
                   model fingerprint in the key.
    editors/       PI (z-space + y-affine — "legacy" reproduces pre-2026-08-31 numbers
                   and is never quoted as PI), ND, GS; plus nullspace + two oracle
                   editors. Editors write; they never score.
    metrics/       arrays in, numbers out, never imports matplotlib. The two Edit Index
                   constructions (ray-zone vs legal-set) share an axis, not a formula —
                   distinct names, quote which one you mean.
    training/      ONE loop, two objectives; the TrainConfig defaults ARE the matched
                   canonical recipe. Entry: scripts/train.py.

Cross-environment comparisons are only meaningful because the architecture, probes,
editors, and training recipe are IDENTICAL across environments up to the input/output
projection and the loss. That invariant is the product; protect it.

## 7. Experiment placement

New experimental code starts as canonical components plus a thin driver — not as a
parallel implementation. A new probe/editor/metric goes into `pim/` with tests and a
REGISTRY row; a new run goes through `scripts/train.py` into a named `runs/<topic>/`;
its scores come from `master_eval.ipynb`. One-off scratch analysis may live in a
notebook, but the moment a number from it is quoted anywhere durable, the computation
moves into `pim/` and the notebook becomes a caller.

## 8. Notebooks

Edit `.ipynb` with the **NotebookEdit** tool; inspect with `Read`/`Grep`. **Never**
manipulate notebook JSON through Bash. Bash is fine for *executing* a notebook
(`nbconvert`) or checking that a file exists. Notebooks are explicit top-to-bottom
pipelines: one operation per cell, no metric math, no hidden pipeline logic in helpers.
