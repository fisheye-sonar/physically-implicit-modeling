# RESEARCH.md — North Star

> **Owner: Sevan (human), by hand, on a slow clock.**
> This is the project's constitution for *what we are trying to learn* — the
> counterpart to CLAUDE.md (*how to work*). Agents **read** this every research
> session and **never write** to it. Its drift over time should be legible in
> `git diff`; that history is a feature. Tactical results do not belong here —
> they go in `research/findings/`. This file changes only when the *question*
> changes.
>
> Status: v1 initial version written by Sevan.

## The question

What is the nature of the **world state** formed by world models when trained primarily to predict observations? We want to understand whether a world state can be found at all, "where it is", what structure that world state has (latent space structure), what affordances matter to us, and under what design constraints and architectures those affordances are achieved. Right now we have four primary affordances, reflected in our evaluation suite:
1. **Predictive Quality.** Does the model produce high quality observations?
2. **Recoverability.** Can we recover the known environment state from the model's internal state?
3. **Coherent Rollout.** Does the model maintain logical coherence when rolling out over many steps, both internally and in the produced observations?
4. **Causal Editability.** Can we intervene and make edits to the model's world state with generative causality?

However, these criteria are a starting point, and we are seeking more unified view of world models. Additionally, we are not trying to come up with an idiosyncratic list to propose to the community, we want to ground our research in first-principles, and genuine, surprising findings. For example, one hypothesized partial unification is that criteria 2-4 are all different forms of identifiability requirements (identifiability of environment state, identifiability of environment dynamics, and causal disentanglement of identifiable factors). It doesn't unify everything, but this simplification will strengthen our research and write-up. Predictive quality can then be folded into a new category called something like observation fidelity.

Another potential unifying theme is that all the affordances may be downstream of a single type of world state: a canonical, factored, predictively-sufficient one (a clean function of the world's minimal sufficient statistic).

The primary thing I care about is *persistence*. I am aiming at world models with persistent world structure, in geometry and in dynamics, over long time horizonts -- not just minutes or seconds.

## "World Models"
The idea of a world model is still undetermined in the field. This research is meant to explore the meaning of the term as much as the properties of it. One long-term aspiration of this project is to expose biases and failure modes of implicit architectures which hint at a desire for explicit physical scaffolding.

For example, we may find that the requirements we demand from a world model's latent space to support the pertinent affordances end up leading to the conclusion that the latent space must have a structure which has strong correspondence to 3D geometry as a theoretical finding. This is just an example. But in that case we would then want to demonstrate how a new architecture can be proposed which has this explicit structure baked in, and the benefits it provides. This is not just diagnostic research, it is diagnostic in service of designing next-era generative world models for stable long-term interaction and consistency.

In later stages of this project, the emphasis on designing and proposing new architectures focused on the merging between implicit trained architectures and explicit physical scaffolding may become far more central.

We should be thinking carefully about the assumptions our design carries with it. These can be sneaky and hard to realize. For example, our choice to treat the environment as an objective, fixed, determined state which our model is attempting to form a representation of is reflective of a traditional objectivist/physicalist mindset. It's possible to explore settings where the world model is a structured system that emerges from the interaction between an uncertain agent and an undetermined environment. This is worth keeping in mind, even if our current project stays focused on a tractable, simplified setup. It is not clear to me yet what kinds of specific results would actually make this framing become more attractive, but I will continue thinking about it and updating the project.

## Three structural sub-questions about the learned state

1. **Geometry — where does it live?**
   Dimensionality, manifold structure, curvature of the set of visited states.
2. **Identifiability — what does it encode?**
   Is the underlying environment state linearly/nonlinearly recoverable by probes?
   Is the recovery stable over long rollouts, or does it decay?
3. **Editability — is it causally manipulable?**
   Do targeted latent interventions produce *coherent, intended* behavioral
   changes (move one object, leave the rest), or do the dynamics reject/distort
   the edit? This is the sharpest interventional test of whether the state is a
   "model" rather than a compression.

Editability is one of the sharper and less explored axes here, so emphasis may be placed on it.

## Scope and stance

- **Substrate:** a controlled toy environment (2D perspective frustum, moving
  discs, 1D intensity scan as the only observation). Chosen because the
  ground-truth latent state is fully known, so probes and edits have a reference.
- **Model under study:** beginning with just a few world models with clean notion of state (GRU, RSSM). Eventually we should include JEPA, Transformers, and diffusion.
- **Epistemic stance:** this research has a strong *diagnostic* component (entirely diagnostic to start). The entire value is in the
  judgment call of whether an effect is real signal or an artifact. We do not
  auto-promote results. "Bug reframed as insight" is the failure mode we most
  guard against — hence the findings-promotion gate (see `research/README.md`).
- **Breadth without losing depth:** multiple directions may run in parallel sessions, but each must connect back to the core questions above. Don't hyperfixate on a single experiment's mechanics at the expense of the larger structural question it serves.

## What would count as answers

- Geometry: a characterization of the visited-state manifold (intrinsic dim,
  curvature) and how it depends on architecture / dataset conditions.
- Identifiability: probe recovery quality + rollout stability, and what training
  conditions produce linearly-recoverable state.
- Editability: a clear account of *when* targeted edits succeed, *when* the
  dynamics reject them, and *why* — separating structural facts about the
  representation from artifacts of the editing method.
