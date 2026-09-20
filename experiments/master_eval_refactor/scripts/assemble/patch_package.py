"""The hand edits on top of build_package.py: notebook globals (RUNS, SETTINGS) become parameters,
and the two loops gain a dry-run guard. Exact-match replacements, each asserted on its hit count."""
import sys
from pathlib import Path

PKG = Path(sys.argv[1]) / "pim" / "scoring"


def patch(name: str, edits: list[tuple[str, str, int]]) -> None:
    p = PKG / name
    t = p.read_text()
    for old, new, count in edits:
        n = t.count(old)
        assert n == count, f"{name}: expected {count} x {old!r}, found {n}"
        t = t.replace(old, new)
    p.write_text(t)
    print(f"patched {name}: {len(edits)} edits")


patch("baselines.py", [
    ("def extra_targets_for(inst, arch, env, s) -> list:",
     "def extra_targets_for(runs, inst, arch, env, s) -> list:", 1),
    ("for r in RUNS:", "for r in runs:", 2),
    ("def score_baselines_arch(inst, env, arch, model_config, s) -> dict:",
     "def score_baselines_arch(runs, inst, env, arch, model_config, s) -> dict:", 1),
    ("extra_targets_for(inst, arch, env)))", "extra_targets_for(runs, inst, arch, env, s), s))", 1),
    ('extra_targets_for(inst, a, spec["env"])', 'extra_targets_for(runs, inst, a, spec["env"], s)', 1),
    ('score_baselines_arch(inst, spec["env"], arch,', 'score_baselines_arch(runs, inst, spec["env"], arch,', 1),
    ('spec["archs"][arch])\n', 'spec["archs"][arch], s)\n', 1),
    ('spec["archs"][arch], ts)', 'spec["archs"][arch], ts, s)', 1),
    ('spec["archs"][arch], bs)', 'spec["archs"][arch], bs, s)', 1),
    ('            print(f"skip  baselines/{inst}  ({BASELINE_VERSION}, archs {sorted(have)})")\n'
     '            continue\n',
     '            print(f"skip  baselines/{inst}  ({BASELINE_VERSION}, archs {sorted(have)})")\n'
     '            continue\n'
     '        if dry_run:\n'
     '            print(f"WOULD fit baselines/{inst}: archs {todo} extra targets {todo_targets} bases {todo_bases}")\n'
     '            todo_all.append({"instance": inst, "archs": todo, "targets": todo_targets, "bases": todo_bases})\n'
     '            continue\n', 1),
])

patch("driver.py", [
    ("import shutil\n\ndef _write_scores", "def _write_scores", 1),      # the cell's mid-file import; shutil is imported at the top
    ("def missing_blocks(r, prev) -> list:", "def missing_blocks(r, prev, s) -> list:", 1),
    ("for k in othello_blocks() if", "for k in othello_blocks(s) if", 1),
    ("""discworld_blocks(f"{r['topic']}/{r['run']}")])""", """discworld_blocks(f"{r['topic']}/{r['run']}", s)])""", 1),
    ("missing = missing_blocks(r, prev)", "missing = missing_blocks(r, prev, s)", 1),
    ('add = scorer(model, r["dir"], only=missing)', 'add = scorer(model, r["dir"], s, only=missing)', 1),
    ("SETTINGS.items()", "s.items()", 2),
    ("done = add_inverse(model, r, prev, missing_im)", "done = add_inverse(model, r, prev, missing_im, s)", 1),
    ('scores = scorer(model, r["dir"])', 'scores = scorer(model, r["dir"], s)', 1),
    ("""                    print(f"skip  {r['topic']}/{r['run']}  (scored at {eval_version(r)})")\n"""
     """                    continue\n""",
     """                    print(f"skip  {r['topic']}/{r['run']}  (scored at {eval_version(r)})")\n"""
     """                    continue\n"""
     """                if dry_run:\n"""
     """                    print(f"WOULD add to {r['topic']}/{r['run']}: blocks {missing}  IM on {missing_inverse(r, prev)}")\n"""
     """                    todo_all.append({"run": f"{r['topic']}/{r['run']}", "action": "add", "blocks": missing,\n"""
     """                                     "inverse": missing_inverse(r, prev)})\n"""
     """                    continue\n""", 1),
    ("""            print(f"skip  {r['topic']}/{r['run']}  (env {r['env']!r} has no scorer)")\n"""
     """            continue\n""",
     """            print(f"skip  {r['topic']}/{r['run']}  (env {r['env']!r} has no scorer)")\n"""
     """            continue\n"""
     """        if dry_run:\n"""
     """            print(f"WOULD score {r['topic']}/{r['run']}  ({'stale' if sp.exists() else 'unscored'})")\n"""
     """            todo_all.append({"run": f"{r['topic']}/{r['run']}", "action": "score"})\n"""
     """            continue\n""", 1),
])

(PKG / "__init__.py").write_text('''"""pim.scoring — the canonical scorer behind ``notebooks/master_eval.ipynb`` (2026-09-19).

The notebook keeps the run scan call, the SETTINGS (every knob, in one visible place) and the
version rule; everything that was wiring, schema or bookkeeping in its cells lives here, moved
verbatim. No metric math: every number is still a call into pim.probes / pim.editors /
pim.metrics / pim.environments.

    runs        which runs are scored (scan_runs; PIM_ONLY_RUNS / PIM_SKIP_TOPICS; still-training guard)
    blocks      which probe-target blocks a run gets, and the scores.json block schema (probe_block)
    discworld   score_discworld (frame models), score_discworld_tokens (frames as tokens)
    othello     score_othello
    baselines   the two decodability floors per (instance, architecture) -> runs/_baselines/
    driver      score_all: score what is missing or stale, ADD blocks / editors a current file lacks
    summary     the per-run human-readable headline
"""
from pim.scoring.baselines import score_all_baselines
from pim.scoring.driver import score_all
from pim.scoring.runs import scan_runs
from pim.scoring.summary import print_summaries

__all__ = ["scan_runs", "score_all_baselines", "score_all", "print_summaries"]
''')
print("wrote __init__.py")
