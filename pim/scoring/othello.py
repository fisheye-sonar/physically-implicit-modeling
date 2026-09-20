"""The Othello scorer (moved verbatim from master_eval cell [4], 2026-09-19). Thin wiring over
pim.environments.othello."""
# [4] The Othello scorer — thin wiring over pim.environments.othello.{arms,bench,corpus}.
#     As on the discworld side, fitted probes live in the run's own probes/ dir.
#     EVERYTHING is mine/theirs, sequence-split (settled 2026-09-01): once the GS
#     target-frame bug was fixed, every editor's best arm read mine/theirs probes, so
#     absolute-colour ("state") probes have no consumer; and the frame split measured
#     0.976 vs sequence's 0.975, so the honest split costs nothing. The grid is 18 fits,
#     down from 72. PROBE_SOURCES still travels into scores.json — one frame is not a
#     reason to stop recording which probes an editor read.
#     ⛔ The INSTANCE travels with the run (2026-09-06): its corpus splits, its RULES
#     (corpus.rules_of: `flip` — oth-noflip never recolours enclosed discs — and, since
#     2026-09-08, `placement` — oth-adjacent places next to an own disc instead of
#     enclosing; replaying games under the wrong rules is an illegal-move assertion) and
#     its intervention bench (Li's shipped 1001 for oth-uniform, the instance's
#     synthesised 1001 otherwise) all come from config.json's data.instance.
#     EXTRA PROBE TARGETS (2026-09-09): SETTINGS["oth_extra_targets"] — the signed
#     mine/theirs REGRESSION target — is scored as its own block under `bases` (the shared
#     `probe_block` shape; the canonical categorical result stays at the top level under
#     the implicit key "mine/theirs"). Its probes are fitted inline (minutes), through the
#     same probe grid; its editors run through the regression branches of
#     `linear_arm` / `grad_steer_arm`; the scorecard, guard and gates are unchanged.
import json

from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import case_targets, load_benchmark
from pim.environments.othello.data import tokens_and_labels, canonical_vocab
from pim.metrics.decodability import probe_skill_from_stats
from pim.metrics.set_editability import move_fidelity_ci95, move_fidelity_ratio
from pim.models import n_points
from pim.probes.mlp import check_probe_sanity
from pim.scoring.blocks import EDITORS_SCORED, IM_VERSION, probe_block

PROBE_SOURCES = {
    "PI": "mine|linear|sequence",
    "ND": "mine|linear|sequence (target−current contrast; see note)",
    "GS": "mine|mlp|sequence (targets: mine/theirs via case_targets)",
}
# ND is the target−current CONTRAST direction (was "ND-sub"), canonical 2026-09-01:
# it beat the plain target-row form on every arm (+0.622 vs +0.447, fid 0.23 vs 0.34)
# and is the more principled direction — it raises the target class while lowering the
# current one, rather than raising the target alone.

def _probe_games(n, instance="oth-uniform"):
    tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s: None, only=("probe",),
                               instance=instance)["probe"])
    itos = {v: k for k, v in canonical_vocab().items()}
    return tokens_and_labels([[itos[int(t)] for t in row[:L]]
                              for row, L in zip(tok[:n], ln[:n])], **oc.rules_of(instance))

def othello_arms(model, bench, lin, mlp, tgt, cur, uns_probs, alphas, s):
    """PI / ND / GS over the bench through ONE probe set (categorical or regression — the
    arms branch on the probe), every arm with the guard attached."""
    a_nd, a_pi, a_gs = alphas
    npnt = n_points(model)
    arms_out = []
    for mode, label, al in (("add_sub", "ND", a_nd), ("pinv", "PI", a_pi)):
        for ell in range(npnt):
            for a in al:
                pr, card = oa.linear_arm(model, bench, lin, tgt, cur, mode=mode, alpha=a, points={ell})
                arms_out.append({"editor": label, "point": ell, "alpha": a,
                                 "fidelity_ratio": move_fidelity_ratio(pr, uns_probs, bench.legal_post),
                                 # the guard's case-level 95% interval (2026-09-19, Sevan) — new fields only,
                                 # as IM / IM-NN and every discworld arm already carry (REGISTRY "case-level spread")
                                 **move_fidelity_ci95(pr, uns_probs, bench.legal_post),
                                 **{k: v for k, v in card.items() if isinstance(v, (int, float))}})
    # GS steers the mine/theirs probes toward mine-coordinate targets. The frames MUST
    # match: feeding absolute-colour labels to these probes is the 2026-08-31 bug, worth
    # 0.70 Edit Index.
    for ls in s["oth_gs_layers"]:
        for a in a_gs:
            pr, card = oa.grad_steer_arm(model, bench, mlp, ls, alpha=a, n_steps=s["oth_gs_steps"],
                                         beta=s["oth_gs_beta"], target_labels=tgt)
            arms_out.append({"editor": "GS", "point": ls, "alpha": a,
                             "fidelity_ratio": move_fidelity_ratio(pr, uns_probs, bench.legal_post),
                             **move_fidelity_ci95(pr, uns_probs, bench.legal_post),
                             **{k: v for k, v in card.items() if isinstance(v, (int, float))}})
    for r in arms_out:                       # the shared block reads `edit_index`
        r["edit_index"] = r["edit_index_union"]
    return arms_out

def score_othello(model, run_dir, s, only=None) -> dict:
    probe_dir = run_dir / "probes"
    inst = (json.loads((run_dir / "config.json").read_text())
            .get("data", {}).get("instance", "oth-uniform"))
    rules = oc.rules_of(inst)
    data = _probe_games(s["oth_probe_games"], inst)
    bench = load_benchmark(inst)
    cur, tgt = case_targets(bench)
    u = oa.unsteered(model, bench)
    uns_probs = oa.unsteered_probs(model, bench)   # the guard's denominator
    npnt = n_points(model)
    out = {"instance": inst, "rules": rules,
           "bench": f"{bench.n_cases} single-tile flips at a fixed 20-move prefix, from {inst}'s own edits games (2026-09-12)",
           "probe_dir": str(probe_dir), "bases": {}}
    # IM / IM-NN (2026-09-15): the state write needs no probe, so ONE computation serves the canonical
    # block and every extra-target block (identical cases); `edit_index` = the union construction, as
    # `othello_arms` sets it, the symmetric difference travels alongside
    _im = {}
    def im_recs():
        if not _im:
            recs, st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=probe_dir,
                                       n_games=s["oth_probe_games"], uns_probs=uns_probs, log=None)
            for r in recs:
                r["edit_index"] = r["edit_index_union"]
            _im["recs"], _im["stats"] = recs, st
            out["inverse_map"] = {"g_r2": st["g_r2"], "g_rmse": st["g_rmse"], "version": IM_VERSION}
        return list(_im["recs"])

    if only is None or "mine/theirs" in only:
        # held-out gates on the full test split (Bayes floors are exact for this generator)
        tok, ln = oc.load(oc.build(oc.LADDER["D"], log=lambda s_: None, only=("test",),
                                   instance=inst)["test"])
        g = oa.gates(model, tok[: s["oth_gates_games"]], ln[: s["oth_gates_games"]], log=None, **rules)
        grid = oa.fit_probe_grid(model, data, cache_dir=probe_dir, log=None)
        # decodability: skill = 1 − err/majority_err, both from the SAME fit's train split
        skill = {}
        for st in grid.stats:
            skill.setdefault((st["target"], st["family"], st["split"]), []).append(probe_skill_from_stats(st))
        lin_mine = {p: grid.probes[("mine", "linear", "sequence", p)] for p in range(npnt)}
        mlp_mine = {p: grid.probes[("mine", "mlp", "sequence", p)] for p in range(npnt)}
        arms_out = othello_arms(model, bench, lin_mine, mlp_mine, tgt, cur, uns_probs,
                                (s["oth_alpha_nd"], s["oth_alpha_pi"], s["oth_alpha_gs"]), s) + im_recs()
        def best(ed):
            sub = [r for r in arms_out if r["editor"] == ed]
            return max(sub, key=lambda r: r["edit_index_union"]) if sub else None
        out |= {
            "gates": g,
            "probe_sources": PROBE_SOURCES,
            "probe_skill": {"|".join(k): v for k, v in skill.items()},
            "probe_stats": [{k: v for k, v in st.items() if not isinstance(v, list)} for st in grid.stats],
            "unedited": {**{k: v for k, v in u.items() if isinstance(v, (int, float))}, "fidelity_ratio": 1.0},
            "best": {ed: best(ed) for ed in EDITORS_SCORED},
            "arms": arms_out,
        }

    for target in s["oth_extra_targets"]:
        if only is not None and target not in only:
            continue
        grid_t = oa.fit_probe_grid(model, data, targets=(target,), cache_dir=probe_dir, log=None)
        lin = {p: (grid_t.probes[(target, "linear", "sequence", p)],
                   next(st for st in grid_t.stats if st["family"] == "linear" and st["point"] == p))
               for p in range(npnt)}
        mlp = {p: (grid_t.probes[(target, "mlp", "sequence", p)],
                   next(st for st in grid_t.stats if st["family"] == "mlp" and st["point"] == p))
               for p in range(npnt)}
        sanity = check_probe_sanity(lin, mlp, strict=False, log=print, label=target)
        lin_p, mlp_p = {p: v[0] for p, v in lin.items()}, {p: v[0] for p, v in mlp.items()}
        alphas = (s["oth_reg_alpha_nd"], s["oth_reg_alpha_pi"], s["oth_reg_alpha_gs"])
        arms_t = othello_arms(model, bench, lin_p, mlp_p, tgt, cur, uns_probs, alphas, s) + im_recs()
        u_t = {**{k: v for k, v in u.items() if isinstance(v, (int, float))},
               "edit_index": u["edit_index_union"], "fidelity_ratio": 1.0}
        block = probe_block(lin, mlp, sanity, u_t, arms_t, ("all",), target=target,
                            basis="mine/theirs", kind="regression", n_classes=None,
                            recipe={"n_games": int(len(data.tokens)), "split": "sequence"},
                            alphas=alphas, selection=None, ei_key="edit_index_union",
                            extra={"probe_sources": {ed: f"{target}|{'mlp' if ed == 'GS' else 'linear'}|sequence"
                                                     for ed in ("PI", "ND", "GS")}})
        out["bases"][target] = block
        print(f"    {target}: skill lin {max(block['probe_skill_linear']):+.4f} mlp "
              f"{max(block['probe_skill_mlp']):+.4f}  " + "  ".join(
                  f"{ed} {b['edit_index_union']:+.3f}/{b['fidelity_ratio']:.2f}"
                  for ed, b in block["best"].items() if b), flush=True)
    return out
