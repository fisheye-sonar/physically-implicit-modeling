"""The two decodability floors per (instance, architecture) (moved verbatim from master_eval
cell [5], 2026-09-19). `score_all_baselines` is the cell's top-level loop."""
# [5] BASELINES — the two decodability floors, per ENVIRONMENT INSTANCE x ARCHITECTURE.
#     They live in runs/_baselines/<instance>/ — the "_" prefix is ALREADY the marker
#     scan_runs and build_full_table use to skip a directory, so baselines sit in runs/
#     without ever being mistaken for a trained run.
#
#     observation  = the SAME probes fitted to the causal INPUT HISTORY instead of the
#                    residual stream. "How much does a shallow read of the input give?"
#     random-init  = the SAME architecture, seeded, never trained, probed identically.
#                    "How much comes from training rather than from random features?"
#     Matched to the model probes in every other respect — corpus, n_seq, the identical
#     seeded 80/20 split BY SEQUENCE, targets, bases, families — so a baseline row and a
#     model row are the same measurement on different features.
#
#     observation_large (b3, 2026-09-02) = the observation floor again on the instance's
#     `probe_large` split (discworld 250k sequences, Othello 170k games — ~5x the rows),
#     50 epochs (>= 2x the canonical step count). The wide-input observation probe
#     MEMORISES the canonical 30k corpus (in-sample gap +0.25 to +0.46 on discworld), so
#     its matched-size floor is an under-estimate; the probe-capacity sweep showed the 5x
#     corpus fixes that (gap ~0.02) and lifts discworld's MLP-128 floor from 0.70 to 0.88.
#     Table 3 reports observation_large where it exists. Model probes are unaffected: at
#     ~14 rows/param they reproduce their 30k values on 250k to within 0.005.
#
#     observation_right / observation_right_large (b4, 2026-09-06) = the same two floors
#     with the history laid out RELATIVE TO THE PRESENT (block 0 = the current frame or
#     move, block k = k steps back). The original left-aligned layout puts the present in a
#     different block for every row, so a LINEAR probe over it cannot express even a
#     current-frame lookup (tokenised dw-8ray: 0.726 left vs 0.973 right; the lookup alone
#     0.968) — the left LIN row understates a linear read of the input. Both layouts are
#     kept; the MLP barely moves between them. Table 3 shows both.
#
#     EXTRA PROBE TARGETS (2026-09-09): a run's extra targets get their two floors too,
#     under the target's own key beside the bases. Discworld (categorical targets):
#     random-init through the ordinary fit_probes path (token inputs for a token
#     architecture) and ONE observation floor, right-aligned, on the target's own recipe
#     (arms.GRID_PROBE_RECIPE → `observation_right_large`); both from probes fitted
#     deliberately outside this notebook (scripts/fit_probes.py; require_cached) — a target
#     whose floors are not fitted yet is SKIPPED and lands on the next run. Othello
#     (the signed regression target): fitted inline like the canonical floors, all four
#     observation layouts. An instance whose baselines.json is current but lacks such a
#     block gets it ADDED.
#
#     ⛔ BOTH floors are per (instance, ARCH), not per instance (fixed 2026-09-01).
#     Random-init obviously so. Observation too, less obviously: the probe is given the
#     history the model actually consumes, and `state_span` is architecture-dependent
#     (transformer_l = block_size 39; transformer_s = n_layers*(window-1)+1). Keying
#     baselines on the instance alone would have silently compared a Transformer-S run
#     against a Transformer-L floor the moment a second architecture was trained on an
#     existing instance — which is exactly what is planned next.
#     Both kinds share ONE probes/ dir per instance: every cache key already carries the
#     model fingerprint (and, for the model-free observation probes, the span), so two
#     architectures cannot collide, and a fit shared between them is computed once.
#     Othello instances carry their RULES (oth-noflip, oth-adjacent): the probe games'
#     labels and the large-split labels are replayed with corpus.rules_of(instance).
import json
import time

import torch

from pim.environments import layout                 # every datasets/ path (layout v2)
from pim.environments.discworld import arms as dwa
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.metrics.decodability import insample_gap_from_stats, probe_skill_from_stats
from pim.models import load_checkpoint
from pim.probes.baselines import random_init_model
from pim.scoring.blocks import by_point, dw_bases_for
from pim.scoring.othello import _probe_games
from pim.scoring.runs import DEV, REPO, _sha

BASELINES_DIR = REPO / "runs" / "_baselines"
BASELINE_SEED = 0
BASELINE_VERSION = "2026-09-06.b4"   # b4 = + right-aligned observation floors (both corpora);
                                     # b3 = + observation_large (5x corpus, 50 epochs);
                                     # b2 = per-(instance, arch); b1 = per-instance.
                                     # Deliberately SEPARATE from EVAL_VERSION: a change
                                     # to the editor sweep must not invalidate a floor,
                                     # and vice versa.
LARGE = {"dw_n_seq": 250_000, "oth_split": "probe_large", "epochs": 50}
# (block name, history layout, epochs — None = the canonical corpus and step count)
OBS_KINDS = (("observation", "left", None), ("observation_large", "left", LARGE["epochs"]),
             ("observation_right", "right", None), ("observation_right_large", "right", LARGE["epochs"]))

def _pack(st):
    """skill + the OVERFIT CHECK. Every fit reports its in-sample counterpart, and
    `insample_gap` is train minus held-out on the skill scale: it is how Table 3 says
    whether a wide-input probe memorised its train split instead of learning the map.
    Model probes sit at +0.000 to +0.005. Both accessors select by the stats' kind
    (R² for regression, 1 − err/majority for classification) and compute nothing new."""
    return {"skill": probe_skill_from_stats(st), "insample_gap": insample_gap_from_stats(st),
            "d_in": st.get("d_in"), "n_train_rows": st.get("n_train_rows")}

def _agg(per_pt):
    """A random-init model has residual points like any other; report its best, the way
    every model row reports its best point. `per_pt` is in residual-point order."""
    best = max(range(len(per_pt)), key=lambda i: per_pt[i]["skill"])
    return {**per_pt[best], "point": best, "per_point": per_pt}

def _dw_encoder(inst, arch):
    """The token encoder for a frames-as-tokens architecture (its probes read TOKEN inputs)."""
    if not arch.endswith("_tokens"):
        return {}
    from pim.environments.discworld import token_bench as tkb
    from pim.environments.discworld.tokens import FrameVocab
    _e, _tag = tkb.token_encoder(FrameVocab.load(layout.tokens_dir(inst) / "vocab.npz"))
    return {"encoder": _e, "encoder_tag": _tag}

def extra_targets_for(runs, inst, arch, env, s) -> list:
    """The extra probe targets of every scored run on this (instance, arch)."""
    if env == "othello":
        return list(s["oth_extra_targets"])
    out = []
    for r in runs:
        if r["instance"] == inst and r["arch"] == arch:
            out += [t for t in s["dw_extra_targets"].get(f"{r['topic']}/{r['run']}", ())
                    if t not in out]
    return out

def _dw_regression_floors(rand, span, inst, target, basis, enc, s) -> dict:
    """The canonical floor set for ONE regression target in ONE basis — the four
    observation layouts / corpora (OBS_KINDS) and the random-init model — fitted inline,
    exactly as `score_baselines_arch` does for the canonical target. Used for the SNAPPED
    regression targets (pos@<partition>, 2026-09-10), whose recipe is the canonical one."""
    pdir = BASELINES_DIR / inst / "probes"
    probe_small, probe_large = {"instance": inst, "size": "120k"}, {"instance": inst, "size": "250k"}
    blk = {k: {} for k, _, _ in OBS_KINDS}
    blk["random_init"] = {}
    for fam in ("linear", "mlp"):
        for kind, align, epochs in OBS_KINDS:
            is_large = epochs is not None
            if is_large and not layout.probe_file("discworld", inst, "250k").exists():
                continue
            _, st = dwa.observation_probes(
                target=target, n_seq=LARGE["dw_n_seq"] if is_large else s["dw_probe_seqs"],
                family=fam, basis_name=basis, span=span,
                probe=probe_large if is_large else probe_small, cache_dir=pdir,
                log=None, epochs=epochs, align=align)
            blk[kind][fam] = {**_pack(st), **({"n_seq": LARGE["dw_n_seq"], "epochs": epochs}
                                               if is_large else {})}
        fits = dwa.fit_probes(rand, target=target, n_seq=s["dw_probe_seqs"], family=fam,
                              basis_name=basis, probe=probe_small, cache_dir=pdir, log=None, **enc)
        blk["random_init"][fam] = _agg([_pack(st) for st in by_point(fits)])
        orr = blk["observation_right_large"].get(fam, {}).get("skill", float("nan"))
        print(f"    {target}/{basis}/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
              f"  obs_right_large {orr:+.4f}  random-init "
              f"{blk['random_init'][fam]['skill']:+.4f}", flush=True)
    return blk

def score_baseline_targets(inst, env, arch, model_config, targets, s) -> dict:
    """Both floors for a run's EXTRA probe targets. Discworld categorical targets: from
    PRE-FITTED probes only (a target without them is skipped); discworld SNAPPED regression
    targets: the canonical floor set, inline. Othello: fitted inline."""
    from pim.environments.discworld.grid_target import categorical_target as _cat
    pdir = BASELINES_DIR / inst / "probes"
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    out = {}
    if env == "discworld":
        basis0 = dw_bases_for(inst, s)[0]      # the instance's probe basis (cartesian on dw-8ray-obs5, 2026-09-14)
        enc = _dw_encoder(inst, arch)
        for target in targets:
            if _cat(target) is None:              # pos@<partition>: a regression target
                out[target] = _dw_regression_floors(rand, span, inst, target, basis0, enc, s)
                continue
            rec = dwa.probe_recipe(target, inst)
            blk = {"random_init": {}, "observation_right_large": {}}
            try:
                for fam in ("linear", "mlp"):
                    _, st = dwa.observation_probes(target=target, family=fam, basis_name=basis0,
                                                   span=span, cache_dir=pdir, log=None, align="right",
                                                   require_cached=True, **rec)
                    blk["observation_right_large"][fam] = {**_pack(st), "n_seq": rec["n_seq"],
                                                           "epochs": rec["epochs"]}
                    fits = dwa.fit_probes(rand, target=target, family=fam, basis_name=basis0,
                                          cache_dir=pdir, log=None, require_cached=True, **rec, **enc)
                    blk["random_init"][fam] = _agg([_pack(st) for st in by_point(fits)])
                    print(f"    {arch}/{target}/{fam}: obs_right_large "
                          f"{blk['observation_right_large'][fam]['skill']:+.4f}  random-init "
                          f"{blk['random_init'][fam]['skill']:+.4f}", flush=True)
            except RuntimeError as e:
                print(f"    {arch}/{target}: floors SKIPPED — {str(e).splitlines()[0][:80]}", flush=True)
                continue
            out[target] = blk
    else:
        rules = oc.rules_of(inst)
        data = _probe_games(s["oth_probe_games"], inst)
        paths = oc.build(only=(LARGE["oth_split"],), log=lambda s_: None, instance=inst)
        data_large = oc.probe_data(paths[LARGE["oth_split"]], **rules)
        for target in targets:
            grid = oa.fit_probe_grid(rand, data, targets=(target,), cache_dir=pdir, log=None)
            blk = {k: {} for k, _, _ in OBS_KINDS}
            blk["random_init"] = {}
            for fam in ("linear", "mlp"):
                for kind, align, epochs in OBS_KINDS:
                    is_large = epochs is not None
                    _, st = oa.observation_probes(data_large if is_large else data, family=fam,
                                                  target=target, seed=BASELINE_SEED, cache_dir=pdir,
                                                  log=None, epochs=epochs, align=align)
                    blk[kind][fam] = {**_pack(st), **({"n_seq": int(len(data_large.tokens)), "epochs": epochs}
                                                       if is_large else {})}
                pts = sorted((x for x in grid.stats if x["target"] == target and x["family"] == fam
                              and x["split"] == "sequence"), key=lambda x: x["point"])
                blk["random_init"][fam] = _agg([_pack(x) for x in pts])
                print(f"    {arch}/{target}/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
                      f"  obs_right_large {blk['observation_right_large'][fam]['skill']:+.4f}"
                      f"  random-init {blk['random_init'][fam]['skill']:+.4f}", flush=True)
            out[target] = blk
    del rand
    return out

def score_baselines_arch(runs, inst, env, arch, model_config, s) -> dict:
    """Both floors for ONE (instance, architecture) pair."""
    pdir = BASELINES_DIR / inst / "probes"
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    block = {"span": span, "bases": {}}
    if env == "discworld":
        # the two probe corpora, by NAME (layout v2): 120k canonical, 250k for the large floors
        probe_small, probe_large = {"instance": inst, "size": "120k"}, {"instance": inst, "size": "250k"}
        # a frames-as-tokens architecture (2026-09-05) is probed on TOKEN inputs — the same
        # probes, corpus and split; only what the model consumes differs (and its cache key)
        enc = _dw_encoder(inst, arch)
        for basis in dw_bases_for(inst, s):
            blk = {k: {} for k, _, _ in OBS_KINDS}
            blk["random_init"] = {}
            for fam in ("linear", "mlp"):
                # observation probes carry NO model; the span is in their cache key, so
                # architectures that share a span share the fit
                for kind, align, epochs in OBS_KINDS:
                    is_large = epochs is not None
                    if is_large and not layout.probe_file("discworld", inst, "250k").exists():
                        continue
                    _, st = dwa.observation_probes(
                        target=s["dw_target"], n_seq=LARGE["dw_n_seq"] if is_large else s["dw_probe_seqs"],
                        family=fam, basis_name=basis, span=span,
                        probe=probe_large if is_large else probe_small, cache_dir=pdir,
                        log=None, epochs=epochs, align=align)
                    blk[kind][fam] = {**_pack(st), **({"n_seq": LARGE["dw_n_seq"], "epochs": epochs}
                                                       if is_large else {})}
                # The random-init floor needs a different MODEL, never a different
                # measurement — this is the ordinary probe path, entirely unchanged.
                fits = dwa.fit_probes(rand, target=s["dw_target"],
                                      n_seq=s["dw_probe_seqs"], family=fam,
                                      basis_name=basis, probe=probe_small,
                                      cache_dir=pdir, log=None, **enc)
                blk["random_init"][fam] = _agg([_pack(st) for st in by_point(fits)])
                ol = blk["observation_large"].get(fam, {}).get("skill", float("nan"))
                orr = blk["observation_right_large"].get(fam, {}).get("skill", float("nan"))
                print(f"    {arch}/{basis}/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
                      f"  obs_large {ol:+.4f}  obs_right_large {orr:+.4f}  random-init "
                      f"{blk['random_init'][fam]['skill']:+.4f}", flush=True)
            block["bases"][basis] = blk
    else:
        rules = oc.rules_of(inst)
        data = _probe_games(s["oth_probe_games"], inst)
        grid = oa.fit_probe_grid(rand, data, cache_dir=pdir, log=None)
        paths = oc.build(only=(LARGE["oth_split"],), log=lambda s_: None, instance=inst)
        data_large = oc.probe_data(paths[LARGE["oth_split"]], **rules)   # labels cached beside the corpus
        blk = {k: {} for k, _, _ in OBS_KINDS}
        blk["random_init"] = {}
        for fam in ("linear", "mlp"):
            for kind, align, epochs in OBS_KINDS:
                is_large = epochs is not None
                _, st = oa.observation_probes(data_large if is_large else data, family=fam,
                                              seed=BASELINE_SEED, cache_dir=pdir, log=None,
                                              epochs=epochs, align=align)
                blk[kind][fam] = {**_pack(st), **({"n_seq": int(len(data_large.tokens)), "epochs": epochs}
                                                   if is_large else {})}
            pts = [x for x in grid.stats if x["target"] == "mine"
                   and x["family"] == fam and x["split"] == "sequence"]
            blk["random_init"][fam] = _agg([_pack(x) for x in pts])
            print(f"    {arch}/mine/{fam}: obs {blk['observation'][fam]['skill']:+.4f}"
                  f"  obs_large {blk['observation_large'][fam]['skill']:+.4f}"
                  f"  obs_right_large {blk['observation_right_large'][fam]['skill']:+.4f}"
                  f"  random-init {blk['random_init'][fam]['skill']:+.4f}", flush=True)
        block["bases"]["mine/theirs"] = blk
    del rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    block["bases"].update(score_baseline_targets(inst, env, arch, model_config,
                                                 extra_targets_for(runs, inst, arch, env, s), s))
    return block

def score_baseline_bases(inst, env, arch, model_config, bases, s) -> dict:
    """The canonical floor set for BASES a CURRENT baselines file lacks (2026-09-15).

    `score_baselines_arch` loops over `dw_bases_for(inst)`, but the loop below only calls it for an
    arch the file does not have — so a basis added to the settings AFTER an arch was first fitted was
    never fitted for it (dw-smooth, dw-16ray: a cartesian run block with frustum-only floors, caught
    when the paper table asked for cartesian). The per-basis body is `_dw_regression_floors` on the
    canonical target, which is exactly what `score_baselines_arch` runs per basis."""
    if env != "discworld" or not bases:
        return {}
    rand = random_init_model(arch, model_config, seed=BASELINE_SEED, device=DEV)
    span = int(getattr(rand, "state_span", 39))
    enc = _dw_encoder(inst, arch)
    out = {b: _dw_regression_floors(rand, span, inst, s["dw_target"], b, enc, s) for b in bases}
    del rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def score_all_baselines(runs, s, dry_run=False) -> list:
    """Fill in every floor a baselines.json lacks, for every (instance, arch) among `runs`.
    `dry_run` reports what WOULD be fitted and fits nothing; returns that to-do list."""
    todo_all = []
    # every (instance, arch) pair that has a trained run — an instance may carry several
    INSTANCES = {}
    for r in runs:
        key = r["instance"]
        if key not in INSTANCES:
            INSTANCES[key] = {"env": r["env"], "archs": {}}
        if r["arch"] not in INSTANCES[key]["archs"]:
            _m, _i = load_checkpoint(r["dir"] / "best_model.pt", device="cpu")
            INSTANCES[key]["archs"][_i.arch] = _i.model_config
            del _m

    for inst, spec in INSTANCES.items():
        bp = BASELINES_DIR / inst / "baselines.json"
        prev = json.loads(bp.read_text()) if bp.exists() else {}
        fresh = prev.get("baseline_version") == BASELINE_VERSION
        have = set(prev.get("archs", {})) if fresh else set()
        todo = [a for a in spec["archs"] if a not in have]
        # extra-target floor blocks a current file lacks
        todo_targets = {a: [t for t in extra_targets_for(runs, inst, a, spec["env"], s)
                            if t not in prev["archs"][a]["bases"]] for a in have} if fresh else {}
        todo_targets = {a: ts for a, ts in todo_targets.items() if ts}
        # regression BASES a current file lacks (2026-09-15) — see score_baseline_bases
        todo_bases = ({a: [b for b in dw_bases_for(inst, s)
                           if b not in prev["archs"][a]["bases"]] for a in have}
                      if fresh and spec["env"] == "discworld" else {})
        todo_bases = {a: bs for a, bs in todo_bases.items() if bs}
        if not todo and not todo_targets and not todo_bases:
            print(f"skip  baselines/{inst}  ({BASELINE_VERSION}, archs {sorted(have)})")
            continue
        if dry_run:
            print(f"WOULD fit baselines/{inst}: archs {todo} extra targets {todo_targets} bases {todo_bases}")
            todo_all.append({"instance": inst, "archs": todo, "targets": todo_targets, "bases": todo_bases})
            continue
        t0 = time.time()
        print(f"\n=== baselines for {inst} ({spec['env']}) archs {todo} extra targets {todo_targets} "
              f"bases {todo_bases} ===", flush=True)
        out = prev if fresh else {"instance": inst, "env": spec["env"],
                                  "seed": BASELINE_SEED, "archs": {}}
        out.setdefault("archs", {})
        for arch in todo:
            out["archs"][arch] = score_baselines_arch(runs, inst, spec["env"], arch,
                                                      spec["archs"][arch], s)
        added = 0
        for arch, ts in todo_targets.items():
            new = score_baseline_targets(inst, spec["env"], arch, spec["archs"][arch], ts, s)
            out["archs"][arch]["bases"].update(new)
            added += len(new)
        for arch, bs in todo_bases.items():
            new = score_baseline_bases(inst, spec["env"], arch, spec["archs"][arch], bs, s)
            out["archs"][arch]["bases"].update(new)
            added += len(new)
        if not todo and not added:
            print(f"    nothing new for {inst} (extra-target floors not fitted yet)")
            continue
        out |= {"baseline_version": BASELINE_VERSION, "commit_sha": _sha(),
                "minutes": round((time.time() - t0) / 60, 1)}
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(json.dumps(out, indent=1, default=float))
        print(f"    wrote {bp.relative_to(REPO)}  [{out['minutes']} min]", flush=True)
    print("\nall baselines present")
    return todo_all
