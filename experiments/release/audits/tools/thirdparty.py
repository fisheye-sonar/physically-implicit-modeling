import ast, sys
sys.argv=[sys.argv[0]]
from closure import ENTRY, source_of, REPO, PIM, mod_name
from collections import defaultdict
stdlib = set(sys.stdlib_module_names)
uses = defaultdict(set)
files = {k: v for k, v in ENTRY.items()}
for p in PIM.rglob("*.py"):
    if "__pycache__" in str(p): continue
    files[mod_name(p)] = p
for k, p in files.items():
    t = ast.parse(source_of(p))
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names: top = a.name.split(".")[0]; uses[top].add(k)
        elif isinstance(n, ast.ImportFrom) and not n.level and n.module:
            uses[n.module.split(".")[0]].add(k)
for top in sorted(uses):
    if top in stdlib or top in ("pim", "__future__"): continue
    print(f"{top:14s} {len(uses[top]):3d}  {', '.join(sorted(uses[top]))[:300]}")
