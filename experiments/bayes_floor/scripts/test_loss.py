#!/usr/bin/env python
"""Table 4 — each run's TEST-set loss beside the instance's estimated Bayes floor (2026-09-12).

    .pim/bin/python experiments/bayes_floor/scripts/test_loss.py            # every scored run
    .pim/bin/python experiments/bayes_floor/scripts/test_loss.py --only L-dw-8ray-20m

Discworld: the training objective (`pim.training.train.mse_next_obs`: teacher-forced next-frame
MSE over every position and ray) on the instance's held-out `eval/test.h5`. The floor is the
STATE-ORACLE prediction — the clean render of the true state advanced one simulator step,
scored against the observed next frame on the same sequences: exactly 0 on a noiseless
instance (the next frame is a deterministic function of the state), the clipped-noise floor
on a noisy one. Othello: the gates already hold the test-split CE and the EXACT Bayes CE
(E[log |legal|]); they are copied here so every run reads from one file.

Writes experiments/bayes_floor/scores/test_loss.json: {run: {env, instance, test_loss, bayes_floor,
excess, n_sequences, note}}. Nothing canonical changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from pim.environments import layout  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402

OUT = _REPO / "experiments" / "bayes_floor" / "scores" / "test_loss.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def counted(rel: Path) -> bool:
    return rel.parts[0] != "archive" and not any(p.startswith("_") for p in rel.parts)


@torch.no_grad()
def dw_test_loss(model, inst: str, batch: int = 256) -> tuple[float, int]:
    from pim.training.train import mse_next_obs

    with h5py.File(layout.eval_file("discworld", inst), "r") as f:
        obs = f["obs_intensity"][:].astype(np.float32)                 # (N, T, R) the noisy frames
    span = int(getattr(model, "state_span", obs.shape[1] - 1))
    x_all = torch.from_numpy(obs[:, : span + 1])                        # the block the loss is defined on
    tot, n = 0.0, 0
    for i in range(0, len(x_all), batch):
        x = x_all[i: i + batch].to(DEV)
        tot += float(mse_next_obs(model, x)) * len(x)
        n += len(x)
    return tot / n, int(len(x_all))


def dw_floor(inst: str) -> tuple[float, str]:
    """State-oracle floor on the test split: render(true state advanced one step) vs the
    observed next frame. 0 exactly when the instance has no noise."""
    from pim.environments.discworld.renderer import render_frame
    from pim.metrics.zone_editability import object_constants, sim_config_from

    with h5py.File(layout.eval_file("discworld", inst), "r") as f:
        sim = json.loads(f.attrs["config_json"])["dataset"]["sim"]
        if float(sim.get("obs_noise_std", 0)) == 0 and float(sim.get("position_noise_std", 0)) == 0:
            if sim.get("blink_prob", 0):
                return 0.0, ("noiseless with blackouts: 0 for a state oracle that knows the blackout schedule; "
                             "the schedule's timing is not inferable from the observations, so the "
                             "history-conditioned floor is above 0 and not estimated here")
            return 0.0, "noiseless: the next frame is a deterministic function of the state"
        n = min(1000, f["obs_intensity"].shape[0])
        obs = f["obs_intensity"][:n].astype(np.float32)
        pos = f["positions"][:n, :, :2, :].astype(np.float32)
        vel = f["velocities"][:n, :, :2, :].astype(np.float32)
    cfg = sim_config_from(sim, 2)
    rad, refl = object_constants(sim, 2)
    dt = float(sim["dt"])
    se, cnt = 0.0, 0
    for i in range(n):
        for t in range(obs.shape[1] - 1):
            r = render_frame((pos[i, t] + vel[i, t] * dt).astype(np.float32), rad, refl, cfg)[2]
            se += float(((r - obs[i, t + 1]) ** 2).sum())
            cnt += r.size
    return se / cnt, f"state oracle (true state + one step, clean render) vs the observed next frame, {n} test sequences"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    floors: dict[str, tuple[float, str]] = {}
    for sp in sorted((_REPO / "runs").rglob("scores.json")):
        rel = sp.relative_to(_REPO / "runs")
        if not counted(rel):
            continue
        s = json.loads(sp.read_text())
        run = s["run"]
        if a.only and sp.parent.name not in a.only:
            continue
        t0 = time.time()
        if s["env"] == "othello":
            g = s["gates"]
            out[run] = {"env": "othello", "instance": s["instance"], "test_loss": g["ce"], "bayes_floor": g["bayes_ce"],
                        "excess": g["ce"] - g["bayes_ce"], "n_sequences": s["settings"].get("oth_gates_games"),
                        "unit": "CE (nats/move)", "note": "gates on the test split; Bayes CE = E[log |legal|], exact"}
        else:
            inst = s["instance"]
            if inst not in floors:
                floors[inst] = dw_floor(inst)
            model, _ = load_checkpoint(sp.parent / "best_model.pt", device=DEV)
            model.eval()
            if s["arch"].endswith("_tokens"):
                # a frames-as-tokens model: its objective is CE over the frame vocabulary; report
                # its own val loss and mark the floor as not comparable (a categorical head has
                # no MSE), rather than invent a bridge here
                out[run] = {"env": "discworld", "instance": inst, "test_loss": s["val_loss"], "bayes_floor": None,
                            "excess": None, "n_sequences": None, "unit": "CE (nats/frame), val split",
                            "note": "token model: CE objective; floor on the frame vocabulary not estimated"}
            else:
                loss, n = dw_test_loss(model, inst)
                fl, note = floors[inst]
                out[run] = {"env": "discworld", "instance": inst, "test_loss": loss, "bayes_floor": fl,
                            "excess": loss - fl, "n_sequences": n, "unit": "MSE (intensity²/ray)", "note": note}
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"{run:<50} {out[run]['test_loss']:.6f}  floor {out[run]['bayes_floor']}  [{time.time() - t0:.0f}s]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"-> {OUT.relative_to(_REPO)}  ({len(out)} runs)")


if __name__ == "__main__":
    main()
