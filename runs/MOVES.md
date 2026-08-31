# runs/ move ledger — housecleaning 2026-08-31
# Rule: NOTHING in runs/ is ever deleted; every move is recorded here.

## Canonical runs (the ONLY two that pass the canonical-runs rule:
## canonical dataset AND canonical architecture AND canonical training setup)
- `scaling/BIG20M_othello_L` → `initial_othello_comparison/L-oth-20m`
- `discworld_scale/BIG20M_discworld_L` → `initial_othello_comparison/L-dw-20m`

## Everything else → archive/ (incl. W16, the S-on-othello rungs, and the L90 pair
## — L90_theirs_discworld trained on dset17's position_noise=0.0 and fails the rule)
- `_review_figures` → `archive/_review_figures`
- `action_editors` → `archive/action_editors`
- `action_sweep` → `archive/action_sweep`
- `controls` → `archive/controls`
- `discworld_scale` → `archive/discworld_scale`
- `dit` → `archive/dit`
- `endogenous` → `archive/endogenous`
- `endogenous_rssm` → `archive/endogenous_rssm`
- `gru` → `archive/gru`
- `gru_multistep` → `archive/gru_multistep`
- `latent_dit` → `archive/latent_dit`
- `latent_linearity` → `archive/latent_linearity`
- `nonlinear_gru` → `archive/nonlinear_gru`
- `omniscient_2d` → `archive/omniscient_2d`
- `othello_arch` → `archive/othello_arch`
- `othello_transfer` → `archive/othello_transfer`
- `ours_on_othello` → `archive/ours_on_othello`
- `rssm` → `archive/rssm`
- `rssm_multistep` → `archive/rssm_multistep`
- `rssm_sweep` → `archive/rssm_sweep`
- `rssm_sweep2` → `archive/rssm_sweep2`
- `scaling` → `archive/scaling`
- `soft_render` → `archive/soft_render`
- `trained_editability` → `archive/trained_editability`
- `transformers` → `archive/transformers`
- `vae` → `archive/vae`

## Kept in place: `_smoke/` (new-scheme pipeline smoke runs), `probe_cache/` (the
## canonical fingerprinted probe cache — no config.json, so the master scan skips it)
