import sys
sys.argv=[sys.argv[0]]
from symbols import run, get_mod
from closure import ENTRY, PIM, mod_name
from collections import defaultdict
reached, reach_by, reached_mods = run(ENTRY)
mods=defaultdict(set)
for k,v in reach_by.items():
    if k[0]!="MOD":
        mods[k[0]] |= v
all_mods = sorted(mod_name(p) for p in PIM.rglob("*.py") if "__pycache__" not in str(p))
short=lambda e: e.split(":")[1]
for m in all_mods:
    e=sorted(mods.get(m,()))
    print(f"{m:44s} {len(e):2d}  {' '.join(short(x) for x in e)}")
