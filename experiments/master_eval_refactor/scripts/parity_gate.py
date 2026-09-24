"""Parity gate: rescore an already-scored run INTO SCRATCH through pim.scoring and diff every
field against the scores.json on disk (and, with --old, against the pre-refactor notebook code
run side by side, which separates "the move changed something" from "the code is not
bit-reproducible against itself" / "pim changed since the file was written").

Nothing is written into runs/: the scorers only read there (probe caches hit; a categorical
target without cached probes is skipped, as in production). The run's probes/ listing is
compared before and after and any new file is reported.

    .pim/bin/python experiments/master_eval_refactor/scripts/parity_gate.py <topic>/<run> [--old]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nb import REPO, old_namespace, settings  # noqa: E402

# fields that legitimately differ between two scorings of one model
VOLATILE = {"minutes", "commit_sha", "blocks_added", "inverse_added", "probe_dir"}
# top-level keys compared APART from the scored numbers, each for a stated reason:
#   settings    a SNAPSHOT of the notebook's SETTINGS at the moment the file was written — it drifts
#               whenever a run is added to dw_extra_targets, without any number changing
#   prediction  folded in by scripts/score_prediction.py (the Bayes-floor work), never by the scorer
APART = ("settings", "prediction")
TOL = 1e-6


def diff(a, b, path="", out=None, stats=None):
    """Every difference between two JSON trees. Floats: exact first, then |Δ| against TOL."""
    out = [] if out is None else out
    stats = {"leaves": 0, "exact": 0, "within_tol": 0, "max_abs": 0.0} if stats is None else stats
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a:
                out.append((f"{path}/{k}", "MISSING in A", None))
            elif k not in b:
                out.append((f"{path}/{k}", "MISSING in B", None))
            else:
                diff(a[k], b[k], f"{path}/{k}", out, stats)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"LENGTH {len(a)} vs {len(b)}", None))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, stats)
    else:
        stats["leaves"] += 1
        def num(v):
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        if num(a) and num(b):
            if a == b or (math.isnan(a) and math.isnan(b)):
                stats["exact"] += 1
            else:
                d = abs(a - b)
                stats["max_abs"] = max(stats["max_abs"], d)
                if d <= TOL:
                    stats["within_tol"] += 1
                else:
                    out.append((path, f"{a!r} vs {b!r}", d))
        elif a == b:
            stats["exact"] += 1
        else:
            out.append((path, f"{a!r} vs {b!r}", None))
    return out, stats


def report(name, a, b, la, lb, fh):
    d, st = diff(a, b)
    missing = [x for x in d if x[1].startswith("MISSING")]
    real = [x for x in d if not x[1].startswith("MISSING")]
    lines = [f"\n=== {name}:  A = {la}   B = {lb} ===",
             f"  leaves compared {st['leaves']} · bit-identical {st['exact']} · within {TOL:g} {st['within_tol']} · "
             f"beyond tolerance {len(real)} · keys on one side only {len(missing)} · max |Δ| {st['max_abs']:.3g}"]
    for p, msg, dd in real[:40]:
        lines.append(f"    DIFF {p}: {msg}" + (f"  (|Δ| {dd:.3g})" if dd is not None else ""))
    if len(real) > 40:
        lines.append(f"    … {len(real) - 40} more")
    for p, msg, _ in missing[:40]:
        lines.append(f"    {msg}: {p}")
    if len(missing) > 40:
        lines.append(f"    … {len(missing) - 40} more one-sided keys")
    for ln in lines:
        print(ln, flush=True)
        fh.write(ln + "\n")
    return len(real), len(missing)


def split(d):
    """(the scored record, the keys compared apart)."""
    return {k: v for k, v in d.items() if k not in APART}, {k: d[k] for k in APART if k in d}


def report_apart(a, b, la, lb, fh):
    lines = []
    sa, sb = a.get("settings", {}), b.get("settings", {})
    d, _ = diff(sa, sb)
    lines.append(f"  settings snapshot ({la} vs {lb}): {len(d)} differing entries"
                 + ("" if not d else " — " + "; ".join(f"{p} [{m if m.startswith('MISSING') else 'changed'}]" for p, m, _ in d[:8])))
    for side, rec in ((la, a), (lb, b)):
        if "prediction" in rec:
            lines.append(f"  `prediction` block present in {side} (written by scripts/score_prediction.py, not by the scorer)")
    for ln in lines:
        print(ln, flush=True)
        fh.write(ln + "\n")


def wrap(scores, r, info, n_pts, s, version):
    """The top-level record exactly as the driver builds it (volatile fields are ignored by diff)."""
    return json.loads(json.dumps(
        {"run": f"{r['topic']}/{r['run']}", "arch": info.arch, "env": r["env"], "instance": r["instance"],
         "val_loss": info.val_loss, "n_points": n_pts, "eval_version": version,
         "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}, **scores},
        default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="<topic>/<run>")
    ap.add_argument("--old", action="store_true", help="also run the pre-refactor notebook code")
    ap.add_argument("--recheck", action="store_true",
                    help="no scoring: re-run the comparison on the JSON a previous gate run saved in outputs/")
    a = ap.parse_args()
    os.chdir(REPO)
    import torch
    from pim.models import load_checkpoint, n_points
    from pim.scoring.driver import scorer_for
    from pim.scoring.runs import DEV, scan_runs

    topic, name = a.run.split("/")
    os.environ["PIM_ONLY_RUNS"] = name
    r = next(x for x in scan_runs() if x["topic"] == topic and x["run"] == name)
    s, eval_version = settings()
    outdir = REPO / "experiments" / "master_eval_refactor" / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = a.run.replace("/", "__")
    probes_before = sorted(p.name for p in (r["dir"] / "probes").glob("*"))
    disk = json.loads((r["dir"] / "scores.json").read_text())

    if a.recheck:
        new = json.loads((outdir / f"{tag}.new.json").read_text())
        old = json.loads((outdir / f"{tag}.old.json").read_text()) if (outdir / f"{tag}.old.json").exists() else None
        t_new = t_old = float("nan")
        return compare(a, tag, r, disk, new, old, t_new, t_old, probes_before, note=" (RECHECK of saved JSON — no scoring)")
    model, info = load_checkpoint(r["dir"] / "best_model.pt", device=DEV)
    scorer, label = scorer_for(r)
    print(f"### NEW code: {scorer.__module__}.{scorer.__name__} on {a.run} ({label}, dw_bases {s['dw_bases']})", flush=True)
    t0 = time.time()
    new = wrap(scorer(model, r["dir"], s), r, info, n_points(model), s, eval_version(r))
    t_new = (time.time() - t0) / 60
    (outdir / f"{tag}.new.json").write_text(json.dumps(new, indent=1))

    old, t_old = None, float("nan")
    if a.old:
        ns = old_namespace()
        old_scorer = ns[scorer.__name__]
        print(f"\n### OLD code: notebook cell function {scorer.__name__} (commit c023401)", flush=True)
        t0 = time.time()
        old = wrap(old_scorer(model, r["dir"]), r, info, n_points(model), ns["SETTINGS"], ns["eval_version"](r))
        t_old = (time.time() - t0) / 60
        (outdir / f"{tag}.old.json").write_text(json.dumps(old, indent=1))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    compare(a, tag, r, disk, new, old, t_new, t_old, probes_before)


def compare(a, tag, r, disk, new, old, t_new, t_old, probes_before, note=""):
    """GREEN iff the move changed nothing: old == new on every scored field (when --old ran), or — without
    --old — the file on disk == new on every scored field; and no probe file appeared in the run dir.
    `settings` / `prediction` are compared apart (see APART) and never decide the verdict. Differences
    against a disk file that the OLD code reproduces identically are reported, not failed: they measure
    drift since that file was written, not the move."""
    scores_dir = REPO / "experiments" / "master_eval_refactor" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    (disk_s, disk_x), (new_s, new_x) = split(disk), split(new)
    with open(scores_dir / f"{tag}.diff.txt", "w") as fh:
        hdr = (f"parity gate — {a.run} — {time.strftime('%F %T')}{note} — new {t_new:.1f} min"
               + (f", old {t_old:.1f} min" if old is not None else "") + f" — disk eval_version {disk.get('eval_version')}")
        print("\n" + hdr, flush=True)
        fh.write(hdr + "\n")
        bad_move = None
        if old is not None:
            old_s, old_x = split(old)
            n_real, n_miss = report("OLD notebook code vs NEW pim.scoring (the move itself)", old_s, new_s, "old", "new", fh)
            report_apart(old_x, new_x, "old", "new", fh)
            bad_move = n_real + n_miss
        n_real, n_miss = report("scores.json ON DISK vs NEW pim.scoring", disk_s, new_s, "disk", "new", fh)
        report_apart(disk_x, new_x, "disk", "new", fh)
        bad_disk = n_real + n_miss
        same_as_old = None
        if old is not None and bad_disk:
            same_as_old = diff(disk_s, split(old)[0])[0] == diff(disk_s, new_s)[0]
            ln = (f"  the OLD code differs from disk in exactly the same {bad_disk} entries: {same_as_old} "
                  f"(True = drift since the file was written, not the move)")
            print(ln, flush=True)
            fh.write(ln + "\n")
        probes_after = sorted(p.name for p in (r["dir"] / "probes").glob("*"))
        extra = sorted(set(probes_after) - set(probes_before))
        ln = f"\n  run's probes/ dir: {len(probes_before)} files before, {len(probes_after)} after; new files: {extra or 'none'}"
        print(ln, flush=True)
        fh.write(ln + "\n")
        ok = (bad_move == 0 and (bad_disk == 0 or same_as_old)) if old is not None else bad_disk == 0
        verdict = "GREEN" if ok and not extra else "RED"
        ln = (f"\n  VERDICT {verdict}: move-induced differences {bad_move if old is not None else 'n/a (no --old)'}; "
              f"scored-field differences against disk {bad_disk}")
        print(ln, flush=True)
        fh.write(ln + "\n")
    sys.exit(0 if verdict == "GREEN" else 1)


if __name__ == "__main__":
    main()
