"""Decision equivalence over the WHOLE runs/ tree: old notebook loops vs pim.scoring.

The pre-refactor notebook's two top-level loops are executed AS WRITTEN (from git) with every
scorer, fitter and writer replaced by a recording stub, so each decision the old code would
take — score this run, add these blocks, add IM here, fit these floors — is captured without
touching a GPU or a file. The refactored loops run with dry_run=True. The two decision lists
must be identical. Exit 0 iff they are.

(A non-empty list is not a failure: a seed replicate inherits its parent's extra targets, and
a target whose probes were never fitted is attempted and skipped on every pass — by design,
"skip-when-uncached". What must hold is that old and new agree on every entry.)
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import OLD_COMMIT, REPO, _cells, old_namespace, settings  # noqa: E402


class _ArchFit(Exception):
    pass


def old_decisions() -> tuple[list, list]:
    ns = old_namespace()
    nb = subprocess.run(["git", "show", f"{OLD_COMMIT}:notebooks/master_eval.ipynb"], cwd=REPO,
                        capture_output=True, text=True, check=True).stdout
    cells = _cells(nb)
    base_loop = cells[5][cells[5].index("# every (instance, arch) pair that has a trained run"):]
    drive_loop = cells[6][cells[6].index("for r in RUNS:"):]

    # ── baselines: the fitters record and return nothing (so the loop never writes) ──
    floors = []
    def _arch(inst, env, arch, cfg, *a, **k):
        floors.append({"instance": inst, "new_arch": arch})
        raise _ArchFit(f"{inst}/{arch}")                  # a whole new arch WOULD be fitted: stop before any write
    ns["score_baselines_arch"] = _arch
    ns["score_baseline_targets"] = lambda inst, env, arch, cfg, ts, *a, **k: (
        floors.append({"instance": inst, "arch": arch, "targets": list(ts)}) or {})
    ns["score_baseline_bases"] = lambda inst, env, arch, cfg, bs, *a, **k: (
        floors.append({"instance": inst, "arch": arch, "bases": list(bs)}) or {})
    try:
        exec(compile(base_loop, "old_master_eval[5-loop]", "exec"), ns)
    except _ArchFit:
        pass

    # ── driver: scorers / IM fold-in / writer record; no checkpoint is loaded ──
    acts = []
    def _scorer(model, run_dir, s=None, only=None):
        key = f"{run_dir.parent.name}/{run_dir.name}"
        acts.append({"run": key, "action": "score"} if only is None else
                    {"run": key, "action": "add", "blocks": list(only)})
        return {"bases": {}}
    for name in ("score_discworld", "score_discworld_tokens", "score_othello"):
        ns[name] = _scorer
    ns["add_inverse"] = lambda model, r, prev, keys, *a, **k: (
        acts.append({"run": f"{r['topic']}/{r['run']}", "action": "inverse", "blocks": list(keys)}) or [])
    ns["_write_scores"] = lambda sp, scores, version: None
    ns["load_checkpoint"] = lambda *a, **k: (None, SimpleNamespace(arch="?", val_loss=None, model_config=None))
    ns["n_points"] = lambda m: 0
    exec(compile(drive_loop, "old_master_eval[6-loop]", "exec"), ns)
    return floors, acts


def new_decisions() -> tuple[list, list]:
    from pim.scoring import scan_runs, score_all, score_all_baselines
    s, eval_version = settings()
    runs = scan_runs()
    floors = []
    for t in score_all_baselines(runs, s, dry_run=True):
        floors += [{"instance": t["instance"], "new_arch": a} for a in t["archs"]]
        floors += [{"instance": t["instance"], "arch": a, "targets": list(ts)} for a, ts in t["targets"].items()]
        floors += [{"instance": t["instance"], "arch": a, "bases": list(bs)} for a, bs in t["bases"].items()]
    acts = []
    for t in score_all(runs, s, eval_version, dry_run=True):
        if t["action"] == "score":
            acts.append({"run": t["run"], "action": "score"})
            continue
        if t["blocks"]:
            acts.append({"run": t["run"], "action": "add", "blocks": list(t["blocks"])})
        if t["inverse"]:
            acts.append({"run": t["run"], "action": "inverse", "blocks": list(t["inverse"])})
    return floors, acts


def main():
    os.chdir(REPO)
    print(f"env: PIM_DW_BASES={os.environ.get('PIM_DW_BASES')!r} PIM_SKIP_TOPICS={os.environ.get('PIM_SKIP_TOPICS')!r} "
          f"PIM_ONLY_RUNS={os.environ.get('PIM_ONLY_RUNS')!r}")
    print("\n################ OLD notebook loops (stubbed) ################", flush=True)
    of, oa_ = old_decisions()
    print("\n################ NEW pim.scoring loops (dry_run) ################", flush=True)
    nf, na = new_decisions()
    def key(d):
        return json.dumps(d, sort_keys=True)

    same_f, same_a = sorted(map(key, of)) == sorted(map(key, nf)), sorted(map(key, oa_)) == sorted(map(key, na))
    out = REPO / "experiments" / "master_eval_refactor" / "scores" / "decisions_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": {k: os.environ.get(k) for k in ("PIM_DW_BASES", "PIM_SKIP_TOPICS", "PIM_ONLY_RUNS")},
                               "old": {"floors": of, "runs": oa_}, "new": {"floors": nf, "runs": na},
                               "identical": {"floors": same_f, "runs": same_a}}, indent=1))
    print(f"\n=== DECISIONS: floors old {len(of)} / new {len(nf)} → identical {same_f} · "
          f"runs old {len(oa_)} / new {len(na)} → identical {same_a} ===")
    n_score = sum(a["action"] == "score" for a in na)
    print(f"    full (re)scores the new driver would start: {n_score}")
    for d in sorted(set(map(key, of)) ^ set(map(key, nf))) + sorted(set(map(key, oa_)) ^ set(map(key, na))):
        print("    ONLY ON ONE SIDE:", d)
    sys.exit(0 if same_f and same_a else 1)


if __name__ == "__main__":
    main()
