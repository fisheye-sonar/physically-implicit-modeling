"""Score every run that needs it, adding only what is missing (moved verbatim from master_eval
cell [6], 2026-09-19). `score_all` is the cell's top-level loop."""
# [6] Score every run whose scores.json is missing or stamped with a stale EVAL_VERSION.
#     (`_sha` comes from [5]; the baselines it wrote are per-instance and already cached.)
#     WHICH SCORER: by what the model EMITS and which world it lives in (2026-09-09) —
#       frame        (regression head; arch without the `_tokens` suffix) on discworld
#                    → score_discworld: ray-zone Edit Index over a rollout
#       distribution (categorical head; `_tokens`) on discworld → score_discworld_tokens:
#                    the frame-set Edit Index at step 0 (the frames-as-tokens runs)
#       distribution on othello → score_othello: the legal-set Edit Index at step 0
#     — the interface is a parameter of the run, never a property of the environment.
#     A run scored at the current version but MISSING a probe-target block the settings
#     ask of it (an extra target added later, 2026-09-09) gets that block ADDED — the
#     existing blocks are not recomputed, and `blocks_added` records when and at what
#     commit each late block landed. A block whose probes are not fitted yet is skipped by
#     the scorer and simply stays missing until they are.
import datetime as _dt
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld import token_bench as tkb
from pim.environments.discworld.tokens import FrameVocab
from pim.environments.othello import arms as oa
from pim.environments.othello import corpus as oc
from pim.environments.othello import load_benchmark
from pim.models import load_checkpoint, n_points
from pim.scoring.blocks import (IM_VERSION, best_arm, cat_inverse_in_scope, discworld_blocks,
                                othello_blocks)
from pim.scoring.discworld import inverse_discworld, score_discworld, score_discworld_tokens
from pim.scoring.othello import _probe_games, score_othello
from pim.scoring.runs import DEV, REPO, _sha

def scorer_for(r) -> tuple:
    emits = "distribution" if r["arch"].endswith("_tokens") else "frame"
    if r["env"] == "othello":
        return score_othello, "othello"
    if emits == "distribution":
        return score_discworld_tokens, "discworld/tokens"
    return score_discworld, "discworld"

SCORED_ENVS = ("discworld", "othello")

def _write_scores(sp: Path, scores: dict, version: str) -> None:
    """Write scores.json ATOMICALLY (tmp → replace) after keeping ONE dated backup of the file as it
    was before the first modification at this eval version (runs/<run>/scores_backup/; 2026-09-15)."""
    if sp.exists():
        bdir = sp.parent / "scores_backup"
        bdir.mkdir(exist_ok=True)
        tag = f"scores_{version}_"
        if not any(p.name.startswith(tag) for p in bdir.glob("scores_*.json")):
            shutil.copy2(sp, bdir / f"{tag}{_dt.date.today().isoformat()}.json")
    tmp = sp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(scores, indent=1, default=float))
    os.replace(tmp, sp)

def _has_im(arms) -> bool:
    return any(a.get("editor") == "IM" for a in (arms or []))

def missing_inverse(r, prev, s=None) -> list:
    """Blocks of a scored run whose arms lack the IM editor (2026-09-15) — the Othello canonical
    block is the top level ("mine/theirs"), every other block sits under `bases`.

    A CATEGORICAL discworld block (2026-09-20) is owed an IM arm only where SETTINGS `dw_cat_im` puts it
    in scope (`blocks.cat_inverse_in_scope`) — everywhere else it has none by design and is never
    "missing". And even in scope it is ADDED to an already-scored run only under `PIM_ADD_CAT_IM=1`:
    each is a ~30-minute streamed fit, so the catch-up over the scored runs is one deliberate job, not
    a surprise inside whichever replicate happens to score next (a run scored FRESH always gets it)."""
    def owed(blk) -> bool:
        if r["env"] != "discworld" or blk.get("kind") != "classification":
            return True
        return (s is not None and os.environ.get("PIM_ADD_CAT_IM") == "1"
                and cat_inverse_in_scope(r.get("instance"), blk.get("target"), s))
    keys = [k for k, blk in prev.get("bases", {}).items() if not _has_im(blk.get("arms")) and owed(blk)]
    if r["env"] == "othello" and "arms" in prev and not _has_im(prev["arms"]):
        keys = ["mine/theirs"] + keys
    return keys

def add_inverse(model, r, prev, keys, s) -> list:
    """Compute IM / IM-NN for the listed blocks of an already-scored run and attach them IN PLACE
    (nothing else in `prev` is touched). Returns the keys done."""
    run_dir = r["dir"]; probe_dir = run_dir / "probes"; inst = r["instance"]
    if r["env"] == "othello":
        rules = oc.rules_of(inst); data = _probe_games(s["oth_probe_games"], inst); bench = load_benchmark(inst)
        uns_probs = oa.unsteered_probs(model, bench)
        recs, st = oa.inverse_arms(model, bench, data, rules=rules, cache_dir=probe_dir,
                                   n_games=s["oth_probe_games"], uns_probs=uns_probs, log=None)
        for x in recs:
            x["edit_index"] = x["edit_index_union"]
        recs = [{k: v for k, v in x.items() if np.isscalar(v)} for x in recs]
        for key in keys:
            blk = prev if key == "mine/theirs" else prev["bases"][key]
            blk["arms"] = [a for a in blk["arms"] if a["editor"] not in ("IM", "IM-NN")] + recs
            for ed in ("IM", "IM-NN"):
                sub = [a for a in blk["arms"] if a["editor"] == ed]
                blk["best"][ed] = best_arm(sub, "edit_index_union") if sub else None
                for d in blk.get("best_by_dims", {}):
                    blk["best_by_dims"][d][ed] = blk["best"][ed]
        prev["inverse_map"] = {"g_r2": st["g_r2"], "g_rmse": st["g_rmse"], "version": IM_VERSION}
        return keys
    tokens = r["arch"].endswith("_tokens")
    benches, ucards, arrays = {}, {}, {}
    if tokens:
        vocab = FrameVocab.load(run_dir / "vocab.npz")
    for key in keys:
        blk = prev["bases"][key]; target, basis = blk["target"], blk["basis"]
        if tokens:
            tb = tkb.load_token_bench(vocab, n=s["dw_bench_n"], target=target, basis_name=basis, instance=inst)
            benches[key], ucards[key] = tb, tkb.unsteered(model, tb)[0]
            arrays[key] = dwb.bench_arrays(s["dw_bench_n"], target, basis, instance=inst)
        else:
            b = dwb.load_bench(model, n=s["dw_bench_n"], target=target, basis_name=basis, instance=inst)
            benches[key], ucards[key] = b, dwa.unsteered(model, b)
    inverse_discworld(model, prev["bases"], benches, ucards, inst, probe_dir, s,
                      tokens=(vocab, arrays) if tokens else None)
    return keys

def missing_blocks(r, prev, s) -> list:
    """Probe-target blocks the settings require of a run but its scores.json lacks. The
    Othello canonical block is the top level (implicit key "mine/theirs")."""
    keys = ([k for k in othello_blocks(s) if k != "mine/theirs"] if r["env"] == "othello"
            else [k for k, _, _ in discworld_blocks(f"{r['topic']}/{r['run']}", s)])
    return [k for k in keys if k not in prev.get("bases", {})]


def score_all(runs, s, eval_version, dry_run=False) -> list:
    """Score every run in `runs` whose scores.json is missing, stale, or lacks a block the
    settings ask of it. `eval_version(r)` is the notebook's version rule. `dry_run` reports
    what WOULD be done and does nothing; returns that to-do list."""
    todo_all = []
    for r in runs:
        sp = r["dir"] / "scores.json"
        if sp.exists():
            prev = json.loads(sp.read_text())
            if prev.get("eval_version") == eval_version(r):
                missing = missing_blocks(r, prev, s)
                if not missing and not missing_inverse(r, prev, s):
                    print(f"skip  {r['topic']}/{r['run']}  (scored at {eval_version(r)})")
                    continue
                if dry_run:
                    print(f"WOULD add to {r['topic']}/{r['run']}: blocks {missing}  IM on {missing_inverse(r, prev, s)}")
                    todo_all.append({"run": f"{r['topic']}/{r['run']}", "action": "add", "blocks": missing,
                                     "inverse": missing_inverse(r, prev, s)})
                    continue
                t0 = time.time()
                scorer, label = scorer_for(r)
                model, info = load_checkpoint(r["dir"] / "best_model.pt", device=DEV)
                if missing:
                    print(f"\n=== adding blocks {missing} to {r['topic']}/{r['run']} ===", flush=True)
                    add = scorer(model, r["dir"], s, only=missing)
                    if not add["bases"]:
                        print(f"    nothing added (probes not fitted yet)", flush=True)
                    else:
                        prev.setdefault("bases", {}).update(add["bases"])
                        prev["settings"] = {k: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in s.items()}
                        prev.setdefault("blocks_added", {}).update(
                            {k: {"date": _dt.date.today().isoformat(), "commit_sha": _sha(),
                                 "minutes": round((time.time() - t0) / 60, 1)} for k in add["bases"]})
                        _write_scores(sp, prev, eval_version(r))
                        print(f"    wrote {sp.relative_to(REPO)}  +{sorted(add['bases'])}  "
                              f"[{round((time.time() - t0) / 60, 1)} min]", flush=True)
                missing_im = missing_inverse(r, prev, s)
                if missing_im:
                    t1 = time.time()
                    print(f"\n=== adding IM / IM-NN to {missing_im} of {r['topic']}/{r['run']} ===", flush=True)
                    n_before = {k: len((prev if k == "mine/theirs" else prev["bases"][k])["arms"]) for k in missing_im}
                    try:
                        done = add_inverse(model, r, prev, missing_im, s)
                    except Exception as e:           # one run's oddity must not kill the chain: say so, move on
                        print(f"    IM SKIPPED for {r['topic']}/{r['run']} — {type(e).__name__}: {str(e).splitlines()[0][:120]}", flush=True)
                        done = []
                    if done:
                        for k in done:      # the fold-in only APPENDS: every pre-existing arm is still there
                            assert len((prev if k == "mine/theirs" else prev["bases"][k])["arms"]) >= n_before[k]
                        prev.setdefault("inverse_added", {}).update(
                            {k: {"date": _dt.date.today().isoformat(), "commit_sha": _sha(), "version": IM_VERSION,
                                 "minutes": round((time.time() - t1) / 60, 1)} for k in done})
                        _write_scores(sp, prev, eval_version(r))
                        print(f"    wrote {sp.relative_to(REPO)}  +IM on {done}  [{round((time.time() - t1) / 60, 1)} min]", flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            print(f"stale {r['topic']}/{r['run']}  ({prev.get('eval_version')} -> {eval_version(r)})")
        if r["env"] not in SCORED_ENVS:
            print(f"skip  {r['topic']}/{r['run']}  (env {r['env']!r} has no scorer)")
            continue
        if dry_run:
            print(f"WOULD score {r['topic']}/{r['run']}  ({'stale' if sp.exists() else 'unscored'})")
            todo_all.append({"run": f"{r['topic']}/{r['run']}", "action": "score"})
            continue
        t0 = time.time()
        scorer, label = scorer_for(r)
        print(f"\n=== scoring {r['topic']}/{r['run']}  ({r['arch']} on {label}) ===", flush=True)
        model, info = load_checkpoint(r["dir"] / "best_model.pt", device=DEV)
        scores = scorer(model, r["dir"], s)
        scores = {"run": f"{r['topic']}/{r['run']}", "arch": info.arch,
                  "env": r["env"], "instance": r["instance"],
                  "val_loss": info.val_loss, "n_points": n_points(model),
                  "eval_version": eval_version(r), "commit_sha": _sha(),
                  "settings": {k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in s.items()},
                  "minutes": round((time.time() - t0) / 60, 1), **scores}
        _write_scores(sp, scores, eval_version(r))
        print(f"    wrote {sp.relative_to(REPO)}  [{scores['minutes']} min]", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nall runs scored")
    return todo_all
