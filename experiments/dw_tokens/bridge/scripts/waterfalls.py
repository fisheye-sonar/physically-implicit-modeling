#!/usr/bin/env python
"""Waterfalls for the frames-as-tokens run, through the canonical `pim.figures.waterfall_grid`.

Columns: GT (prepended by the figure), the unedited free-run, PI's best arm and GS's best
arm (from scores/bridge_<run>.json), each its OWN rollout; signed-error columns beside them.
Two figures per basis: the EXPECTED-frame rendering (what the bridge scores) and the ARGMAX
rendering (crisp frames), both with argmax feedback. Rows are `random_samples` (seeded),
never the largest teleports. Locators: the mean ray index of the target / ghost zones.

Output: outputs/waterfall_<run>_<basis>_<render>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import TokenFrameAdapter  # noqa: E402
from score_bridge import EXP, SETTINGS, cached_probes  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.figures.waterfall import waterfall_grid  # noqa: E402
from pim.metrics.zone_editability import random_samples  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402


def locator(mask: np.ndarray) -> np.ndarray:
    """(N, R) bool -> (N,) mean ray index of the masked rays (NaN where empty)."""
    R = mask.shape[1]
    w = mask.astype(float)
    with np.errstate(invalid="ignore"):
        return (w * np.arange(R)[None, :]).sum(1) / w.sum(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/interface_ablation/L-dw-8ray-tok-20m")
    ap.add_argument("--bases", nargs="+", default=["frustum", "cartesian"])
    ap.add_argument("--k", type=int, default=4, help="rows per figure")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    s = dict(SETTINGS)
    tag = "_smoke" if a.smoke else ""
    if a.smoke:
        s.update({"dw_probe_seqs": 400, "dw_bench_n": 24, "dw_gs_steps": 5})
    run_dir = (REPO / a.run).resolve()
    S = json.loads((EXP / "scores" / f"bridge_{run_dir.name}{tag}.json").read_text())
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = REPO / "datasets" / "discworld" / cfg["data"]["instance"]
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=dwb.DEV)
    model.eval()
    vocab = FrameVocab.load(run_dir / "vocab.npz")
    ad = TokenFrameAdapter(model, vocab, render="expected", feedback="argmax").eval()
    (EXP / "outputs").mkdir(exist_ok=True)
    for basis in a.bases:
        if basis not in S["bases"]:
            continue
        b = dwb.load_bench(ad, n=s["dw_bench_n"], target=s["dw_target"], basis_name=basis, data_dir=inst / "eval")
        lin = cached_probes(model, run_dir, inst, basis, "linear", s["dw_probe_seqs"], vocab)
        mlp = cached_probes(model, run_dir, inst, basis, "mlp", s["dw_probe_seqs"], vocab)
        best = S["bases"][basis]["best"]
        rows = random_samples(b.n, a.k, 0)
        tx, gx = locator(b.zones.target), locator(b.zones.ghost)
        for render in ("expected", "argmax"):
            ad.render = render
            roll_u = dwa.unsteered_rollout(ad, b)
            u = dwa.score(ad, b, roll_u)
            pi, gs = best["PI"], best["GS"]
            roll_pi = dwa.pinv_rollout(ad, b, lin[pi["point"]][0], pi["point"], pi["alpha"], space="zspace", dims=pi["dims"])
            roll_gs = dwa.grad_steer_rollout(ad, b, mlp, gs["point"], gs["alpha"], n_steps=s["dw_gs_steps"],
                                             beta=s["dw_gs_beta"], dims=gs["dims"])
            cols = {"unedited": roll_u,
                    f"PI {pi['dims']}·pt{pi['point']}·α{pi['alpha']:g}": roll_pi,
                    f"GS {gs['dims']}·pt{gs['point']}·α{gs['alpha']:g}": roll_gs}
            metrics = {name: dwa.score(ad, b, r, u)["edit_index"] for name, r in cols.items()}
            fig = waterfall_grid(
                cols, b.obs[:, :dwb.EF], b.gt_roll,
                title=(f"{run_dir.name} · {basis} basis · {render}-frame rendering, argmax feedback · "
                       f"edit at frame {dwb.EF}, {dwb.K_ROLL}-step free-run · rows = random_samples(seed 0)"),
                diff_columns={f"{n} − GT": r - b.gt_roll for n, r in cols.items()}, diff_scale=0.8,
                sample_idx=rows, target_x=tx, ghost_x=gx, metrics=metrics)
            out = EXP / "outputs" / f"waterfall_{run_dir.name}_{basis}_{render}{tag}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print("wrote", out, {k: round(v, 3) for k, v in metrics.items()}, flush=True)
        ad.render = "expected"


if __name__ == "__main__":
    main()
