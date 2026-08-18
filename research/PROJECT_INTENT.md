# PROJECT_INTENT.md — the long-horizon destination

*Stated by Sevan 2026-08-04. Migrated here from auto-memory 2026-08-17.*

> **This is further out than anything in `RESEARCH.md`.** `RESEARCH.md` is human-owned and
> agents never write it, so this intent is recorded here instead — it is the destination the
> current diagnostic work is serving, not a direction brief. If it ever belongs in the vision
> file, that is Sevan's edit to make.

## The direction: integrate world models into AI systems

Not world models as an object of study only, but as a **component wired to a language model's
output**. Three properties:

1. **Grounded report.** Ask "what do you see" and the generated text is *accurate to the world
   model* — the text is a readout of a state, not a free invention.
2. **Grounded action.** Ask it to move an object; it says it did; **the movement is visible in
   the world model**.
3. **Efference-copy verification.** Follow up with "did your action succeed?" — and because the
   world state is known independently, the answer can be **checked**, not merely asserted.

## Why the current work feeds this

The editability thread is establishing what a latent world state can and cannot support:
`readable ≠ grabbable` (probe-directed writes do not move objects), the Δh reachability ceiling,
the Edit Index. **Point 2 above is exactly the editability question** — *can an external agent
write to the world state and have the dynamics honour it* — asked of a system where the writer
is a language model.

The 2026-08-14 trained-editor result (`findings/trained-editors.md`) is the first mechanism to
answer that affirmatively, and its shape is informative: the editor needed to be told the
*start* as well as the target, i.e. it needed the displacement rather than having to infer the
world from the state.

## The load-bearing design constraint

*(Claude's, 2026-08-04, and worth keeping.)* The verification in (3) is only well-posed when
**the state and the report are separable**. If the only record of the world is the text the
model emits, "did it succeed?" has no independent answer and **cannot be wrong**. The world
model must be something the generator does *not* author.

## Substrate that already exists here

- `pim/simulator/interactive.py` (`InteractiveWorld`) — a stateful `reset`/`step` world **not
  authored by the model**.
- `scripts/play.py` — an agent drives it; a keyboard overlay shows the model's actions; the
  **real** observation waterfall sits beside the model's **predicted** one.
- `AutoregressiveModelDriver` — the "dreaming" mode where the model consumes its own
  predictions. This is the efference-copy failure made visible: deaths rise from 2.8 to ~85 per
  1000 frames closed-loop.
