import json, sys
from pathlib import Path
import torch
REPO = Path("/home/sevan/research/PIM/physically-implicit-modeling")
sys.path.insert(0, str(REPO))
from pim.models.registry import build
from pim.probes.cache import fingerprint
R = json.load(open(Path(__file__).parent / "inventory_runs.json"))
for r in R:
    rc = r.get("replicate_cfg") or {}
    if not rc.get("checkpoint"):
        continue
    src = REPO / rc["source"]
    if not src.exists():
        print(r["run"], "source missing", src); continue
    ck = torch.load(src, map_location="cpu", weights_only=False)
    rep = torch.load(REPO / "runs" / r["run"] / "best_model.pt", map_location="cpu", weights_only=False)
    m = build(r["arch"], dict(rep.get("model_config") or {}))
    m.load_state_dict(ck["model_state"])
    same = all(torch.equal(ck["model_state"][k], rep["model_state"][k]) for k in ck["model_state"])
    print(f"{r['run']:<62} fp(src ckpt)={fingerprint(m)} fp(replicate)={r['fingerprint']} state_equal={same} src_keys={sorted(ck)} diff_keys={sorted(set(rep)-set(ck))}")
