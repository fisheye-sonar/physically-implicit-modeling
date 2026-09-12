"""Lay out a canonical run's saved checkpoint as a SEED-0 REPLICATE at that step budget
(``runs/<topic>/<run>__seed0_s<step>/``) so the replicate set pools seed 0 at a MATCHED
training budget with the half-step seeds. Same construction as the training-curve view
(experiments/training_curve/scripts/make_training_curve.py): the checkpoint is copied, given
the nearest validated val_loss, and stamped with a `replicate` block.
    python experiments/seed_variance/scripts/layout_checkpoint_replicate.py noise_ablation/L-dw-noiseless-20m 421875
"""
import json
import shutil
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
parent, step = sys.argv[1], int(sys.argv[2])
src = REPO / "runs" / parent
dst = src.parent / f"{src.name}__seed0_s{step}"
if (dst / "best_model.pt").exists():
    print("exists", dst.relative_to(REPO)); sys.exit(0)
vals = {int(x["step"]): float(x["val_loss"]) for x in map(json.loads, open(src / "metrics.jsonl")) if x.get("val_loss") is not None}
cfg = json.loads((src / "config.json").read_text())
ck = torch.load(src / "ckpt" / f"step_{step:09d}.pt", map_location="cpu", weights_only=False)
near = min(vals, key=lambda v: abs(v - step)); ck["val_loss"], ck["val_loss_step"] = vals[near], near; ck["arch"] = cfg["arch"]
dst.mkdir(parents=True, exist_ok=True); torch.save(ck, dst / "best_model.pt")
c = dict(cfg)
c["replicate"] = {"of": parent, "seed": int(cfg["train"].get("seed", 0)), "steps": step, "checkpoint": True,
                  "source": f"runs/{parent}/ckpt/step_{step:09d}.pt", "val_loss_from_step": near,
                  "note": "the canonical run's own checkpoint at the replicates' step budget (2026-09-11)"}
(dst / "config.json").write_text(json.dumps(c, indent=1))
if (src / "commit_sha").exists():
    shutil.copy(src / "commit_sha", dst / "commit_sha")
print("laid out", dst.relative_to(REPO), "val", ck["val_loss"], "from step", near)
