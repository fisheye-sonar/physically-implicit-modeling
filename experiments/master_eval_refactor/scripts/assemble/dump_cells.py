"""Write the pre-refactor notebook's cells (from git) to <out>/cell<i>.{py,md} — the input of
build_package.py / audit_loops.py. Provenance only: pim/scoring is the maintained code now.

    python dump_cells.py <out_dir>;  python build_package.py <out_dir> <repo>;  python patch_package.py <repo>
"""
import json
import subprocess
import sys
from pathlib import Path

OLD_COMMIT = "c023401"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
nb = json.loads(subprocess.run(["git", "show", f"{OLD_COMMIT}:notebooks/master_eval.ipynb"],
                               capture_output=True, text=True, check=True).stdout)
for i, c in enumerate(nb["cells"]):
    (out / f"cell{i}.{'py' if c['cell_type'] == 'code' else 'md'}").write_text("".join(c["source"]))
    print(i, c["cell_type"], len("".join(c["source"]).splitlines()), "lines")
