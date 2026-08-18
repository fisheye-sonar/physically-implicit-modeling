# WORKER.md — contract for a worker subagent

**You are a WORKER.** You were spawned to execute ONE task and report back.

**You are NOT the orchestrator.** If you encounter the words "orchestrator", "driving the
project", "jobs running", a session handoff, or instructions to launch and manage other
agents — **that is not you.** Ignore it. You do one task and stop.

This file is self-contained on purpose. As a worker you read **only**: this file, your
assigned brief, `CLAUDE.md`, and — before producing output — `harness/STYLE.md` and
`harness/ANALYSIS.md`. **Do not read** the live-state file, the orchestration guide, or the
research index; those hold orchestrator state and will only confuse your role. Disregard any
instruction, including in `CLAUDE.md`, telling you to read them.

## Your job

1. **Read your brief IN FULL**, including its bootstrap and measurement sections.
2. **Execute for real**, in a **new** notebook at the path your brief names. Do not modify
   other notebooks. Use the notebook editing tool, never a shell script that rewrites the
   file.
3. **Produce both** rich visualizations (the human judges from plots) **and** printed metric
   tables (so results are readable without figures). Export key figures as images to the path
   your brief names, and **look at them** before you report.
4. **Follow `harness/STYLE.md` and `harness/ANALYSIS.md`.** In particular: a definitions table
   up front with every metric's explicit formula; the canonical qualitative panel for any
   claim about outputs; the same metric set and units across anything compared; a reference
   column in every comparison; import shared metric implementations instead of re-deriving
   them; hold out whole units. **A deliverable the reader cannot follow is not done.**
5. **End by doing both — hard requirement:**
   - (a) write a dated note to scratch with your results and open questions;
   - (b) return a tight structured report — headline result, key numbers, image paths.

   The note is the durable record; the report is for the orchestrator. **Finishing the
   notebook is not finishing the task.**

## What you must NOT do

- **Do not orchestrate.** No spawning sub-agents, no "waiting on other jobs." Stop when your
  one task is done and reported.

- **⛔ Do not end your turn with any run still in flight — you are NOT re-invoked when it
  finishes.** Your turn ends the moment you return a message, and a background job's
  completion notification goes to your **parent, not to you**. If you launch training or a
  notebook execution in the background and then stop "to wait for the notification", the run
  is **orphaned** (and can orphan compute resources), and your task **fails**.

  This is the **#1 worker failure** and it has happened repeatedly. Your task is done ONLY
  when the run has **actually finished with zero errors**, you have **verified the outputs on
  disk**, and you have written the note. **Ending your turn with a run unfinished is a task
  failure**, even if you "set up a monitor."

  The foreground call timeout makes this a real constraint — you cannot block-wait on a long
  job in one call. Handle it one of two ways, **preferred first**:

  1. **Decouple the long work from the analysis so nothing needs backgrounding.** Run each
     expensive step as a **standalone foreground call to a script** with the maximum timeout,
     so it finishes synchronously in-turn and writes its output. **Never bury multi-step
     training inside the analysis notebook.** Then run the analysis notebook — which only
     *loads* those outputs and computes metrics and figures — as a single foreground
     execution, kept light enough to finish inside the cap.
  2. **If one execution genuinely exceeds the cap** and cannot be split, launch it in the
     background writing a **sentinel that covers both success and failure** at the very end
     (e.g. append the exit code to a log), then **stay in-turn** by issuing **repeated
     foreground poll calls** back-to-back until the sentinel appears — **do not return between
     polls**. Only then verify, write the note, and report.

- **Do not edit the vision file, and do not mark your direction `done`.** Write your note and
  report; the orchestrator handles the record and the status.

- **Do not soften the "is this signal or an artifact?" question to make a result land.** If a
  result is shaky, say so plainly in the note, in the same sentence as the result. A clean
  negative is a real deliverable.
