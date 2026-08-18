# UPSTREAM.md — portable lessons, and what porting taught us

Two logs. Both exist so that carrying the harness to another project is **one read** rather
than a re-derivation from a large project-specific file.

---

## 1. Portable lessons awaiting integration

When you learn something that would be true in a different project, write it here **in the
same session**, even if it is one line and not yet well-phrased. Periodically fold entries
into the appropriate harness file and mark them integrated.

The bar: *would this be true in a project with different data, a different domain, and
different models?* If it needs a domain noun to state, it is not portable — it belongs in the
project's own gotchas file.

| date | lesson | destination | status |
|---|---|---|---|
| 2026-08-17 | Prose does not prevent re-implementation drift; a single imported implementation does. Enforcement hierarchy: shared code > point-of-action checklist > hook > prose. | `ANALYSIS.md` §1 | integrated |
| 2026-08-17 | Hold out whole units, never sub-units — and the damage is worst for labels that are constant within a unit, because the model can identify the unit instead of decoding the quantity. | `ANALYSIS.md` §2 | integrated |
| 2026-08-17 | A single-step report of a horizon metric can state the opposite of the truth; mechanisms invert over a horizon. | `ANALYSIS.md` §3 | integrated |
| 2026-08-17 | An approval-gated record becomes a dam: measured here as one month of zero updates while 25 notes queued behind it. Grade confidence on the entry instead of gating the write. | `WORKFLOW.md` | integrated |
| 2026-08-17 | Agents cannot see figures; humans cannot read printed tables at volume. Every deliverable needs both, always. | `STYLE.md` | integrated |
| 2026-08-17 | Role confusion between orchestrator and worker is a real failure mode with silent consequences (work completed, never reported). Defend it twice: in the auto-loaded project file *and* in the launch prompt. | `ORCHESTRATION.md`, `WORKER.md` | integrated |
| 2026-08-17 | A vocabulary deny-list will over-match ordinary English — "steering" fired on "research steering" within minutes of being written. Budget for narrowing patterns, and record *why* each narrowing was made, or the list silently rots into permissiveness. | `check.sh` comments | integrated |
| 2026-08-17 | Enforce a spec by making its violations **unrepresentable in the API**, not by documenting them. The canonical panel helper takes one rollout *per column*, so the banned shared row cannot be passed; scaling limits are parameters with fixed defaults, so per-cell autoscaling cannot happen by accident. Strictly stronger than a checklist. | `ANALYSIS.md` §1, `STYLE.md` §2 | integrated |
| 2026-08-17 | Rendering a figure and *looking at it* caught two defects (two semantically different markers in near-identical colours; a reversed axis-direction label) that reading the code did not. "It ran without error" really is not the check. | `STYLE.md` §3 | integrated |

---

## 2. Port record — what changed on the way to another project

**This is the whole reason the file exists.** The quarantine check catches project *nouns*;
it cannot catch a rule that is phrased generically but is only true in one domain. Those leak
silently, and **an actual port is the only reliable test.**

So: when this harness is copied into a new project, record **every edit made to make it fit**.
That diff is the measured list of rules that were secretly local. It is also the evidence base
for deciding whether a shared, synchronized harness repository is worth building — a decision
that should be made from data, not from a guess.

| date | target project | file | what had to change | verdict |
|---|---|---|---|---|
| — | *(none yet — first port pending)* | | | |

**Verdict vocabulary:** `local` (was never universal — move it out of the harness),
`generalize` (universal but badly phrased — rewrite it to be domain-free),
`extend` (both projects need it, in different variants — state the variants).

### Staging plan

- **Stage 1 (done):** build `harness/` in this project, designed for portability from the
  start — universal prose, pointer sections, quarantine check.
- **Stage 2 (next port):** copy it into the new project. Fill in the port record above. Do
  **not** attempt to synchronize the two copies yet.
- **Stage 3 (conditional):** if the port record shows divergence is small and the same rules
  keep being re-learned independently, promote the harness to a standalone repository with a
  real sync mechanism. If divergence is large, the correct conclusion is that these are two
  local harnesses that share an ancestor, and that is fine.

Do not skip to stage 3. With only one project of evidence, "universal" is a guess, and a
wrong universal rule propagates to every project at once carrying borrowed authority.
