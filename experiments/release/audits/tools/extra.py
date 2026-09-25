import sys
sys.argv=[sys.argv[0]]
from symbols import run
from closure import REPO, PIM, mod_name
from collections import defaultdict
from pathlib import Path
E={}
for p in ["paper/figs/predictive_quality/othello.py", "paper/figs/predictive_quality/rayworld.py",
          "paper/figs/predictive_quality/compute_rayworld.py", "paper/figs/editability_trends/by_point.py",
          "paper/figs/editability_trends/by_rays.py", "paper/figs/history_rewrite/make_figure.py",
          "paper/figs/history_rewrite/draw_paper.py"]+[str(p.relative_to(REPO)) for p in (REPO/"paper/figs/environments_overview").rglob("*.py")]+[str(p.relative_to(REPO)) for p in (REPO/"experiments/paper_ci/scripts").glob("*.py")]:
    if (REPO/p).exists(): E[p]=REPO/p
reached, reach_by, _ = run(E)
mods=defaultdict(lambda: defaultdict(set))
for k,v in reach_by.items():
    if k[0]!="MOD":
        for e in v: mods[e][k[0]].add(k[1])
for e in E:
    print("==", e)
    for m in sorted(mods[e]):
        print("   ", m, sorted(mods[e][m])[:12], "..." if len(mods[e][m])>12 else "")
