"""ORACLE editor 1 — counterfactual state overwrite: replace the state wholesale.

Not an editor anyone could deploy (it needs the simulator to produce the post-edit
world's observations); it is the CEILING every real editor is read against, and one of
the two oracle arms kept to defend the Edit Index itself: if the metric can score well
under this edit and freeze-time interpolation, then a workhorse editor scoring at the
unedited floor is a fact about the model, not about the measure.

The edit: render the post-edit world's observation history (the edits split stores it
as ``clean_obs`` / can be re-rendered with the dataset's noise), build the state the
model would have had if it had SEEN that world, and continue from it:

    state_cf = model.state_from_obs(cf_frames)      # protocol call, any architecture
    rollout  = free-run from state_cf, no edit applied

For window/prefix-state models this is exact by construction. It writes every
dimension of the carried state, which is what makes it the upper bound on "the model's
dynamics can carry the edited world forward at all".

⚠ Noise-matching matters: a model trained on noisy observations warmed on CLEAN
counterfactual frames is mildly out of distribution. Pass the frames you would have
teacher-forced — the dataset's noisy observations — unless the clean-frame arm is the
deliberate comparison.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def counterfactual_state(model, cf_frames: torch.Tensor):
    """The state the model would carry had it observed ``cf_frames`` (B, T, obs_res)."""
    return model.state_from_obs(cf_frames)


@torch.no_grad()
def overwrite_rollout(model, cf_frames: torch.Tensor, steps: int) -> torch.Tensor:
    """(B, steps, obs_res) free-run from the counterfactual state, no edit applied."""
    s = counterfactual_state(model, cf_frames)
    pred = model.decode(s)
    out, s = [pred], model.advance(s, pred)
    for _ in range(steps - 1):
        p, s = model.predict_step(s)
        out.append(p)
    return torch.stack(out, 1)
