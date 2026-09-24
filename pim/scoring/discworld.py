"""The discworld scorers — frame models and frames-as-tokens models (moved verbatim from
master_eval cells [3] / [3b], 2026-09-19). Thin wiring over pim.environments.discworld."""
# [3b] The discworld TOKEN scorer (2026-09-05) — thin wiring over
#      pim.environments.discworld.token_bench: a frames-as-tokens model (the Othello
#      architecture on the instance's frame vocabulary) scored the OTHELLO way.
#      Same instance, probe corpus, regression probes (LIN / MLP-128 on the full state,
#      both bases) and editors (PI z-space, GS; ND on categorical targets) as
#      score_discworld. The model's output is a DISTRIBUTION over next frames, so the Edit
#      Index is Othello's legal-set construction on the two worlds' frames at the edit
#      frame ("frame-set", reported under `edit_index`: +1 = the edited world's frame, −1 =
#      the unedited one), the guard is `move_fidelity_ratio`, and `p_post` is the mass on
#      the edited world's frame. The block schema is the shared `probe_block`, so
#      build_full_table reads it unchanged; `ei_construction` marks the difference, and the
#      bridge to the ray-zone construction (`zone_edit_index_expected`) rides along, never
#      as the headline. Extra (categorical) targets are scored as their own blocks exactly
#      as in score_discworld — probes on TOKEN inputs, the categorical MOVE as the edit.
#      ⛔ FILTERED CASE SET (2026-09-08). The bench is NOT the first 192 cases: on dw-8ray
#      17% of teleports render an IDENTICAL frame (mean teleport 0.92 world units against
#      2.42 for the rest) and another 22% move a single ray of eight, so a first-192 bench
#      scored only 163 cases, many of them marginal. `token_bench.load_token_bench` now
#      reads the instance's `edits_selection.json` — the first 192 cases whose two worlds
#      differ on >= 2 rays with both frames in the vocabulary — same generator, same seeds,
#      same split, only which cases are scored (192/192 scoreable, mean teleport 2.82). The
#      selection travels into scores.json as `bench_selection` so a number can always be
#      traced to the case list that produced it.
import json

from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld import token_bench as tkb
from pim.environments.discworld.tokens import FrameVocab
from pim.models import n_points
from pim.probes.mlp import check_probe_sanity
from pim.probes.inverse import CATEGORICAL_STATE
from pim.scoring.blocks import (attach_inverse, cat_inverse_in_scope, discworld_blocks, dw_block_setup,
                                probe_block)
from pim.scoring.runs import REPO


def inverse_discworld(model, blocks: dict, benches: dict, ucards: dict, inst: str, probe_dir, s,
                      tokens=None) -> None:
    """IM / IM-NN for the blocks in ``benches`` ({key: Bench}); ``ucards`` = {key: unsteered card}
    (frames) or {key: unsteered probs} (tokens; ``tokens=(vocab, arrays)``).

    The inverse map inverts THE STATE THE BLOCK'S OWN PROBES READ (2026-09-20, Sevan):
    * REGRESSION blocks — one map per point per BASIS from the continuous full state (cached in the
      run's probes/, kind inverse_map), IM and IM-NN, as since 2026-09-15;
    * CATEGORICAL blocks — the target's own one-hot labels + the discs' Cartesian velocity, fitted with
      the target's forward-probe recipe (``cat_inverse_in_scope``: SETTINGS ``dw_cat_im`` names the
      instances and targets; elsewhere a categorical block carries NO IM arm). IM only.
    Until 2026-09-20 a categorical block was handed its basis's CONTINUOUS map — a state its probes
    never read — and those arms were removed from every scores.json."""
    reg = [k for k in benches if blocks[k]["kind"] == "regression"]
    for k in (k for k in benches if blocks[k]["kind"] != "regression"):
        if not cat_inverse_in_scope(inst, blocks[k]["target"], s):
            continue
        recipe = dwa.probe_recipe(blocks[k]["target"], inst, n_seq=s["dw_probe_seqs"])
        if tokens is None:
            arms, st = dwa.inverse_arms(model, {k: benches[k]}, basis_name=blocks[k]["basis"],
                                        target=blocks[k]["target"], unsteered_cards={k: ucards[k]},
                                        cache_dir=probe_dir, log=None, **recipe)
        else:
            vocab, arrays = tokens
            arms, st = tkb.inverse_arms(model, {k: benches[k]}, {k: arrays[k]}, vocab,
                                        basis_name=blocks[k]["basis"], target=blocks[k]["target"],
                                        uns={k: ucards[k]}, cache_dir=probe_dir, log=None, **recipe)
        attach_inverse(blocks, arms, st, ei_key="edit_index")
        blocks[k]["inverse_map"].update({"state": CATEGORICAL_STATE, "epochs": recipe["epochs"],
                                         "n_seq": recipe["n_seq"], "g_r2_insample": st.get("g_r2_insample")})
        im = blocks[k]["best"]["IM"]
        print(f"    {k}: IM[categorical state] {im['edit_index']:+.4f}/{im['fidelity_ratio']:.2f} (pt {im['point']})  "
              f"g R² max {max(st['g_r2']):+.3f}", flush=True)
    recipe = dwa.probe_recipe("full", inst, n_seq=s["dw_probe_seqs"])
    for basis in sorted({blocks[k]["basis"] for k in reg}):
        keys = [k for k in reg if blocks[k]["basis"] == basis]
        if tokens is None:
            arms, st = dwa.inverse_arms(model, {k: benches[k] for k in keys}, basis_name=basis,
                                        unsteered_cards={k: ucards[k] for k in keys}, cache_dir=probe_dir,
                                        log=None, **recipe)
        else:
            vocab, arrays = tokens
            arms, st = tkb.inverse_arms(model, {k: benches[k] for k in keys}, {k: arrays[k] for k in keys}, vocab,
                                        basis_name=basis, uns={k: ucards[k] for k in keys}, cache_dir=probe_dir,
                                        log=None, **recipe)
        attach_inverse(blocks, arms, st, ei_key="edit_index")
        for k in keys:
            im = blocks[k]["best"]["IM"]
            print(f"    {k}: IM {im['edit_index']:+.4f}/{im['fidelity_ratio']:.2f} (pt {im['point']})  "
                  f"IM-NN {blocks[k]['best']['IM-NN']['edit_index']:+.4f}  g R² max {max(st['g_r2']):+.3f}", flush=True)

def score_discworld(model, run_dir, s, only=None) -> dict:
    """Every block of `discworld_blocks` (or just the keys in `only`, when adding to an
    existing scores.json)."""
    probe_dir = run_dir / "probes"
    run_key = f"{run_dir.parent.name}/{run_dir.name}"
    inst = (json.loads((run_dir / "config.json").read_text())
            .get("data", {}).get("instance", "dw-pn04"))
    out = {"probe_dir": str(probe_dir), "instance": inst, "target": s["dw_target"],
           "edit_dims": list(s["dw_edit_dims"]), "bases": {}}
    benches, ucards = {}, {}
    for key, target, basis in discworld_blocks(run_key, s):
        if only is not None and key not in only:
            continue
        cat, dimsets, (a_nd, a_pi, a_gs) = dw_block_setup(target, s)
        recipe = dwa.probe_recipe(target, inst, n_seq=s["dw_probe_seqs"])
        try:
            lin = dwa.fit_probes(model, target=target, family="linear", basis_name=basis,
                                 cache_dir=probe_dir, log=None, require_cached=cat is not None,
                                 **recipe)
            mlp = dwa.fit_probes(model, target=target, family="mlp", basis_name=basis,
                                 cache_dir=probe_dir, log=None, require_cached=cat is not None,
                                 **recipe)
        except RuntimeError as e:
            print(f"    {key}: SKIPPED — {str(e).splitlines()[0][:90]}", flush=True)
            continue
        b = dwb.load_bench(model, n=s["dw_bench_n"], target=target,
                           basis_name=basis, instance=inst)
        sanity = check_probe_sanity(lin, mlp, strict=False, log=print, label=key)
        u = dwa.unsteered(model, b)
        arms = []
        for dims in dimsets:
            for ell in range(n_points(model)):
                arms += dwa.nanda_arm(model, b, lin[ell][0], ell, a_nd, dims=dims)
            arms += dwa.pinv_arm(model, b, lin, a_pi, space="zspace", dims=dims)
            arms += dwa.grad_steer_arm(model, b, mlp, s["gs_layers"], a_gs,
                                       n_steps=s["dw_gs_steps"], beta=s["dw_gs_beta"],
                                       dims=dims)
        for r in arms:
            r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)
        block = probe_block(lin, mlp, sanity, u, arms, dimsets, target=target, basis=basis,
                            kind=b.kind, n_classes=next(iter(lin.values()))[0].n_classes if cat else None,
                            recipe=recipe, alphas=(a_nd, a_pi, a_gs), selection=b.selection)
        out["bases"][key] = block
        benches[key], ucards[key] = b, u
        pi = block["best"]["PI"]
        print(f"    {key}: skill lin {max(block['probe_skill_linear']):+.4f}"
              f"  PI {pi['edit_index']:+.4f}/{pi['fidelity_ratio']:.2f}"
              f" (dims={pi['dims']})", flush=True)
    if benches:
        inverse_discworld(model, out["bases"], benches, ucards, inst, probe_dir, s)
    return out

def score_discworld_tokens(model, run_dir, s, only=None) -> dict:
    probe_dir = run_dir / "probes"
    run_key = f"{run_dir.parent.name}/{run_dir.name}"
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg.get("data", {}).get("instance", "dw-8ray")
    vocab = FrameVocab.load(run_dir / "vocab.npz")        # the run's OWN vocabulary
    enc, tag = tkb.token_encoder(vocab)
    sel_path = tkb.selection_path(instance=inst)          # edits/v1/selection.json (layout v2)
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else None
    out = {"probe_dir": str(probe_dir), "instance": inst, "target": s["dw_target"],
           "edit_dims": list(s["dw_edit_dims"]), "repr": "tokens", "vocab_size": int(vocab.size),
           "ei_construction": "frame-set",
           "bench_selection": ({"file": str(sel_path.relative_to(REPO)), "rule": sel["rule"],
                                "n": sel["n"], "min_rays": sel["min_rays"],
                                "pool": sel["pool"], "stats": sel["stats"]}
                               if sel else "first-n (no selection file)"),
           "bases": {}}
    benches, ucards, arrays = {}, {}, {}
    for key, target, basis in discworld_blocks(run_key, s):
        if only is not None and key not in only:
            continue
        cat, dimsets, (a_nd, a_pi, a_gs) = dw_block_setup(target, s)
        recipe = dwa.probe_recipe(target, inst, n_seq=s["dw_probe_seqs"])
        try:
            lin = dwa.fit_probes(model, target=target, family="linear", basis_name=basis,
                                 cache_dir=probe_dir, log=None, encoder=enc, encoder_tag=tag,
                                 require_cached=cat is not None, **recipe)
            mlp = dwa.fit_probes(model, target=target, family="mlp", basis_name=basis,
                                 cache_dir=probe_dir, log=None, encoder=enc, encoder_tag=tag,
                                 require_cached=cat is not None, **recipe)
        except RuntimeError as e:
            print(f"    {key}: SKIPPED — {str(e).splitlines()[0][:90]}", flush=True)
            continue
        tb = tkb.load_token_bench(vocab, n=s["dw_bench_n"], target=target,
                                  basis_name=basis, instance=inst)
        sanity = check_probe_sanity(lin, mlp, strict=False, log=print, label=key)
        uns, u = tkb.unsteered(model, tb)
        arms = []
        for dims in dimsets:
            if cat is not None:                      # ND: categorical targets only
                for ell in range(n_points(model)):
                    arms += tkb.nanda_arm(model, tb, lin[ell][0], ell, a_nd, uns, dims=dims)
            arms += tkb.pinv_arm(model, tb, lin, a_pi, uns, space="zspace", dims=dims)
            arms += tkb.grad_steer_arm(model, tb, mlp, s["gs_layers"], a_gs, uns,
                                       n_steps=s["dw_gs_steps"], beta=s["dw_gs_beta"], dims=dims)
        block = probe_block(lin, mlp, sanity, u, arms, dimsets, target=target, basis=basis,
                            kind=tb.kind, n_classes=next(iter(lin.values()))[0].n_classes if cat else None,
                            recipe=recipe, alphas=(a_nd, a_pi, a_gs),
                            selection=tb.selection if tb.selection else out["bench_selection"],
                            extra={"n_cases_kept": int(tb.keep.sum())})   # pre != post frame
        out["bases"][key] = block
        benches[key], ucards[key] = tb, uns
        arrays[key] = dwb.bench_arrays(s["dw_bench_n"], target, basis, instance=inst)   # the same cases' world state
        pi = block["best"]["PI"]
        print(f"    {key}: skill lin {max(block['probe_skill_linear']):+.4f}"
              f"  unedited {block['unedited']['edit_index']:+.4f}"
              f"  PI {pi['edit_index']:+.4f}/{pi['fidelity_ratio']:.2f} (dims={pi['dims']})"
              f"  cases kept {block['n_cases_kept']}/{tb.n}", flush=True)
    if benches:
        inverse_discworld(model, out["bases"], benches, ucards, inst, probe_dir, s, tokens=(vocab, arrays))
    return out
