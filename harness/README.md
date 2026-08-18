# harness/ — portable working standards

How to do the work well: how results are presented, how claims are checked, how the
research record is kept, and how orchestrators and workers divide the job.

**This directory is deliberately project-agnostic.** Everything here is intended to be
copied wholesale into a new project and be immediately useful. What is *specific to this
project* lives elsewhere — see the routing table below. The quarantine is enforced, not
merely intended: `harness/check.sh` fails if project vocabulary appears in a harness file
outside its `Local instantiations` section.

## Read these, by role

**Every session, whatever your role:** the project's `CLAUDE.md` (auto-loaded) tells you
which role you are. Then:

| Role | Read |
|---|---|
| **Orchestrator** — a session started by the human. This is the default. | `COLLABORATION.md`, `WORKFLOW.md`, `ORCHESTRATION.md`, then the project's live state file |
| **Worker** — every spawned subagent, without exception | `WORKER.md` and your assigned brief. **Nothing else** from the research record. |

Then, triggered by what you are about to do:

| About to… | Read first |
|---|---|
| write any figure, table, notebook, or reader-facing deliverable | `STYLE.md` |
| compute a metric, fit anything, or make an empirical claim | `ANALYSIS.md` |
| write up a result, update the record, or decide what is established | `WORKFLOW.md` |
| spawn a subagent | `ORCHESTRATION.md` |
| start a new project or a new experiment thread | `templates/` (rarely — not part of any session ritual) |

## Where things go — routing

Four buckets. Every piece of written knowledge belongs to exactly one, and the boundaries
are structural rather than a matter of judgment:

| Content | Home | Portable? |
|---|---|---|
| Universal rules — how to work well anywhere | `harness/*.md` | **yes** |
| Pointers to this project's instantiations of those rules | the `Local instantiations` section at the foot of each harness file | no — deleted on port |
| Project mechanics — commands, environment, paths, architecture | `CLAUDE.md` | no — rewritten per project |
| Project traps — landmines, stale conventions, "check this first" | the project's gotchas file | no |
| Science — what is true, what is next, raw observation | the research record (`research/`) | no |
| Concrete local specs — metric registries, run registries, panel specs | beside the work they govern, in the experiments tree | no |

The last row matters and is easy to get wrong. A concrete spec (the exact parameters of a
standard estimator, the exact layout of this project's comparison panel) **does not belong
in this directory**, even when the rule requiring it does. Harness states the rule; the
project states the instantiation; harness links to it by a one-line pointer. This is the
whole quarantine strategy, and it exists because duplicated specs drift, and a drifted copy
inside `harness/` is worse than no copy — it carries borrowed authority.

## The quarantine rule

**A harness file that names a project-specific noun, outside its `Local instantiations`
section, is a bug.** Model architectures, dataset names, run codes, domain objects, metric
names invented here — none of them. If a rule needs a concrete example to be usable, the
example goes at the pointer target, not inline.

Run `bash harness/check.sh` to verify. It also fires automatically on any write into
`harness/`, so a violation surfaces at the moment it is made rather than at port time.

**Known limit, stated honestly:** the check catches project *nouns*. It cannot catch a rule
that is phrased generically but is only true here — an assumption about the shape of the
data, a threshold that made sense in one domain. Those leak silently, and the only reliable
test is an actual port to a different project. That is why `UPSTREAM.md` exists.

## Extending the harness

- A rule earns its place by having **cost something**. Prefer rules traceable to a real
  failure over rules that merely sound wise. Keep a one-clause "why" — it is what makes a
  rule stick — and drop the incident narrative, which does not travel.
- When you learn something portable, add it to `UPSTREAM.md` in the same session. That file
  is the export manifest for the next project; if it is current, porting is one read instead
  of a re-derivation from scratch.
- When a new project noun enters the vocabulary, add it to the deny-list in `check.sh`.
- Keep files short enough to actually be read. A rule nobody reads is not a rule.
