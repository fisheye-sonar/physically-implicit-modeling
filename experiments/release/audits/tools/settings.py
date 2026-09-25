"""Read-only: rebuild master_eval SETTINGS / eval_version by exec'ing the notebook's settings cells."""
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/sevan/research/PIM/physically-implicit-modeling")
sys.path.insert(0, str(REPO))


def load_settings():
    nb = json.loads((REPO / "notebooks" / "master_eval.ipynb").read_text())
    ns = {"os": os}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith("# [2] The canonical") or src.startswith("# [2b] CATEGORICAL"):
            exec(src, ns)
    return ns["SETTINGS"], ns["eval_version"], ns["EVAL_VERSION_BY_ENV"], ns["EVAL_VERSION"]


if __name__ == "__main__":
    s, ev, evb, e0 = load_settings()
    print(evb, e0)
    print(sorted(s))
    print(s["dw_bases"], s["dw_cat_im"])
