"""Build the training-curve view: one run dir per log-spaced checkpoint of a canonical run.

Each `runs/training_curve/<run>_s<step>/` is a real trained model (the canonical run's
own checkpoint at that step) laid out as a run dir, so `master_eval.ipynb` scores it
through the identical canonical path — no second scorer. The checkpoint is COPIED
(never linked) because the intermediate files store `val_loss: None`, which
`load_checkpoint` cannot float; the copy carries the nearest validated val_loss from
metrics.jsonl and says so in config.json.

Question it answers: does editability EMERGE with training after decodability has
already saturated (Othello), and does discworld's ever move (or trend) at all?
"""
import json
import shutil
from pathlib import Path

import torch

REPO = Path.cwd()
STEPS = [1000, 4000, 16000, 64000, 128000, 256000, 512000, 780000]
RUNS = {"L-dw-20m": "initial_othello_comparison", "L-oth-20m": "initial_othello_comparison"}

for run, topic in RUNS.items():
    src = REPO / "runs" / topic / run
    vals = {}
    for line in open(src / "metrics.jsonl"):
        x = json.loads(line)
        if x.get("val_loss") is not None:
            vals[int(x["step"])] = float(x["val_loss"])
    cfg = json.loads((src / "config.json").read_text())
    for s in STEPS:
        dst = REPO / "runs" / "training_curve" / f"{run}_s{s:06d}"
        if (dst / "best_model.pt").exists():
            # Repair pass, never a refit: the pre-housecleaning Othello trainer stamped its
            # intermediate checkpoints arch="theirs" (the vendored minGPT), which the
            # registry does not know. Normalise an existing copy in place.
            ck = torch.load(dst / "best_model.pt", map_location="cpu", weights_only=False)
            if ck.get("arch") != cfg["arch"]:
                ck["arch"] = cfg["arch"]
                torch.save(ck, dst / "best_model.pt")
                print("patched arch ->", cfg["arch"], dst.name, flush=True)
            else:
                print("exists", dst.name, flush=True)
            continue
        ck_path = src / "ckpt" / f"step_{s:09d}.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        near = min(vals, key=lambda v: abs(v - s))
        ck["val_loss"], ck["val_loss_step"] = vals[near], near
        ck["arch"] = cfg["arch"]        # normalise legacy names to the registry's
        dst.mkdir(parents=True, exist_ok=True)
        torch.save(ck, dst / "best_model.pt")
        c = dict(cfg)
        c["curve"] = {"source_run": f"{topic}/{run}", "step": s,
                      "checkpoint": str(ck_path.relative_to(REPO)),
                      "val_loss_from_step": near,
                      "note": "training-curve view of the canonical run's own checkpoint; "
                              "val_loss is the nearest validated step in metrics.jsonl"}
        (dst / "config.json").write_text(json.dumps(c, indent=1))
        if (src / "commit_sha").exists():
            shutil.copy(src / "commit_sha", dst / "commit_sha")
        print(f"built {dst.name}  val {vals[near]:.5f} (from step {near})", flush=True)
print("curve dirs ready")
