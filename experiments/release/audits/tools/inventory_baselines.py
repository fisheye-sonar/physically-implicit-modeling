"""READ-ONLY inventory of runs/_baselines/<instance>/ for the in-scope instances."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
from settings import REPO, load_settings  # noqa: E402

sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.baselines import random_init_model  # noqa: E402
from pim.probes.cache import fingerprint  # noqa: E402

S, eval_version, _, _ = load_settings()
INST = ["oth-uniform", "oth-adjacent-flip", "oth-adjacent", "oth-noflip", "dw-noiseless", "dw-blink", "dw-128ray",
        "dw-16ray", "dw-8ray", "dw-5ray", "dw-smooth", "dw-8ray-obs5"]
# one run per (instance, arch) to get the model_config
RUN_FOR = {
    ("oth-uniform", "transformer_l_tokens"): "initial_othello_comparison/L-oth-20m",
    ("oth-adjacent-flip", "transformer_l_tokens"): "adjacent_flip_ablation/L-oth-adjacent-flip-20m",
    ("oth-adjacent", "transformer_l_tokens"): "adjacency_ablation/L-oth-adjacent-20m",
    ("oth-noflip", "transformer_l_tokens"): "flip_ablation/L-oth-noflip-20m",
    ("dw-noiseless", "transformer_l"): "noise_ablation/L-dw-noiseless-20m",
    ("dw-blink", "transformer_l"): "blink_ablation/L-dw-blink-20m",
    ("dw-128ray", "transformer_l"): "ray_ablation/L-dw-128ray-20m",
    ("dw-16ray", "transformer_l"): "ray_ablation/L-dw-16ray-20m",
    ("dw-8ray", "transformer_l"): "ray_ablation/L-dw-8ray-20m",
    ("dw-8ray", "transformer_l_tokens"): "interface_ablation/L-dw-8ray-tok-20m",
    ("dw-5ray", "transformer_l"): "ray_ablation/L-dw-5ray-20m",
    ("dw-smooth", "transformer_l"): "smooth_ablation/L-dw-smooth-20m",
    ("dw-8ray-obs5", "transformer_l"): "observer_ablation/L-dw-8ray-obs5-20m",
}

rand_fp = {}
for (inst, arch), rk in RUN_FOR.items():
    _m, info = load_checkpoint(REPO / "runs" / rk / "best_model.pt", device="cpu")
    rm = random_init_model(info.arch, info.model_config, seed=0, device="cpu")
    rand_fp[(inst, info.arch)] = fingerprint(rm)
    del _m, rm

out = {}
for inst in INST:
    d = REPO / "runs" / "_baselines" / inst
    rec = {"files": {}, "probes": []}
    for p in sorted(d.iterdir()):
        if p.is_file():
            rec["files"][p.name] = p.stat().st_size
    bj = json.loads((d / "baselines.json").read_text())
    rec["baseline_version"] = bj.get("baseline_version")
    rec["archs"] = {a: sorted(v.get("bases", {})) for a, v in bj.get("archs", {}).items()}
    fps = {v: k for k, v in rand_fp.items() if k[0] == inst}
    for p in sorted((d / "probes").iterdir()):
        e = {"file": p.name, "bytes": p.stat().st_size}
        if p.suffix == ".pt":
            prov = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["provenance"]
            e["prov"] = prov
            m = prov.get("model")
            if m == "none":
                e["cat"] = "observation"
            elif m in fps:
                e["cat"] = f"random-init[{fps[m][1]}]"
            else:
                e["cat"] = f"random-init[fp {m} not reproduced]"
            # what the tables read (tables._rand_perdim): random-init, target full, any basis/family
            e["tables_read"] = (m != "none" and prov.get("target") == "full")
        else:
            e["cat"] = "index" if p.name == "INDEX.md" else "other"
        rec["probes"].append(e)
    out[inst] = rec
    agg = defaultdict(lambda: [0, 0])
    for e in rec["probes"]:
        pv = e.get("prov", {})
        k = (e["cat"], pv.get("kind", "-"), pv.get("target", "-"), pv.get("basis", "-"),
             pv.get("align", "-"), pv.get("epochs", "-"), pv.get("n_seq", "-"), pv.get("split", "-"),
             "enc" if pv.get("encoder") else "", pv.get("family", pv.get("families", "-")) if False else "",
             e.get("tables_read", False))
        agg[k][0] += 1
        agg[k][1] += e["bytes"]
    print(f"=== {inst}  version {rec['baseline_version']}  archs {rec['archs']}")
    print("   files:", rec["files"])
    for k, (n, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"   {n:>3} {b/1e6:9.2f} MB  {k}")
(SCR / "inventory_baselines.json").write_text(json.dumps(out, indent=1, default=str))
print({f"{k[0]}|{k[1]}": v for k, v in rand_fp.items()})
