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
| write any figure, table, notebook, or deliverable | `harness/STYLE.md` |
| compute a metric, fit anything, make an empirical claim | `harness/ANALYSIS.md` |
| touch data, metrics, or an old number | `research/GOTCHAS.md` |
| write up a result or update the record | `harness/WORKFLOW.md` |
| spawn a subagent | `harness/ORCHESTRATION.md` |
| build the canonical qualitative panel | `notebooks/experiments/editability/WATERFALL_SPEC.md` (1D) · `.../omniscient_2d/WATERFALL_SPEC_2D.md` (2D) |
| use or define a metric or editor | `notebooks/experiments/editability/METRICS_AND_EDITORS.md` |

**Two hard rules restated here because they are violated most often:**
- **Any claim about an effect on the model's generations ships with a waterfall** — built
  through the one shared `waterfall_grid(...)` helper, per the spec above. A scorecard hides
  the difference between "the edit landed" and "the output degraded".
- **Import shared metric implementations; never re-derive them.**
  `scripts/editability_metrics.py` and `pim.extractors.fit_readability_probes` are the single
  source. A genuinely new metric is fine — add its registry row in the same commit.

## 3. Commands

```bash
# tests
poetry run pytest

# demo animation (interactive)
python scripts/demo.py --seed 7 --n-objects 4 --waterfall-mode human

# dataset (splits: train/val/test/edits)
python scripts/generate_dataset.py data/my_run --n-train 100000 --n-workers 8

# train
python scripts/train_gru.py
python scripts/train_rssm.py

# eval (auto-detects model type; writes figures + metrics.json + eval_config.json)
python scripts/run_eval.py <checkpoint>

# lint / format
poetry run ruff check pim tests
poetry run black pim tests scripts

# harness quarantine check (after editing anything in harness/)
bash harness/check.sh
```

## 4. Environment

- Python 3.13 venv in `.pim/`; [direnv](https://direnv.net/) auto-activates via `.envrc`
  (else `source .pim/bin/activate`). Dependencies via Poetry.
- **Run in the main working tree, not a git worktree** — `datasets/` and `runs/` are
  gitignored, so a worktree has no data or checkpoints to load.
- `.claude/settings.local.json` sets `worktree.bgIsolation: "none"` and allows
  `Write`/`Edit`/`NotebookEdit`, so background workers can edit the main tree directly.

## 5. Where things live

- Datasets: `datasets/<n>_<name>/` (train/val/test/edits HDF5 + `dataset.json`)
- Checkpoints: `runs/<thread>/<code>/ckpt_final.pt` (gitignored)
- Eval outputs: timestamped dirs under `outputs/`
- Experiments: `notebooks/experiments/<thread>/<name>.ipynb`
- Run registries: the `*_RUNS.md` file in each thread directory

## 6. Architecture — the contracts that matter

Five strictly separated layers: `simulator/` (ground-truth world, rendering, datasets) →
`world_models/` (architectures behind a shared Protocol) → `extractors/` (probes) →
`editors/` (interventions) → `eval/` (metrics) and `figures/` (drawing). The per-module map is
deliberately not duplicated here — it goes stale; use `Grep`/`Read` to navigate.

Four invariants you cannot infer from any single file:

1. **eval/figures purity.** `pim/eval/*` returns arrays and scalars and **never imports
   matplotlib**. `pim/figures/*` takes pre-computed arrays and returns a `Figure`, and never
   calls models or computes metrics. The notebook or script is the orchestrator that wires
   them. Changing *how a metric is computed* → `eval/`; *how it is drawn* → `figures/`. Do not
   collapse these.
2. **Model-agnostic eval via Protocol.** All eval flows through the `HiddenStateModel`
   protocol (`flat_state` / `state_from_flat` / `decode` / `observe_sequence` /
   `predict_step`). A new architecture implements the protocol and the whole suite works
   unchanged — never add `isinstance` branches in `eval/`.
3. **Probe uniformity.** Any number of probes flow as `list[ProbeSpec]` through `fit_probes`
   → `eval_recovery_multi` → figure builders. Add a probe by instantiating a `ProbeSpec`.
   Probe hyperparameters live in the extractor constructor, never in `EvalConfig`.
4. **Notebook = orchestrator.** Eval notebooks are explicit top-to-bottom pipelines: one
   operation per cell, passing named artifacts to the next. Keep them linear; do not hide
   pipeline logic in helpers.

## 7. Experiment placement

**Every experiment goes inside the research thread it serves — never beside it.** The topic
directory is the *findings thread* the work contributes to, not the kind of experiment it is.
While the editability findings are the active thread, **everything** — main experiments,
controls, ablations, side quests — lives under `notebooks/experiments/editability/`, in a
subdirectory when it is a coherent group. Creating a sibling like
`notebooks/experiments/controls/` orphans the work from the thread whose findings it supports
and breaks every relative path back to the thread's registries.
*(This happened on 2026-07-30 with the michael_controls thread and had to be migrated.)*

Run artifacts follow the same logic: a thread's checkpoints and eval outputs stay together
under one `runs/<thread>/` prefix.

## 8. Notebooks

Edit `.ipynb` with the **NotebookEdit** tool; inspect with `Read`/`Grep`. **Never** manipulate
notebook JSON through Bash. Bash is fine for *executing* a notebook (`nbconvert`) or checking
that a file exists.
