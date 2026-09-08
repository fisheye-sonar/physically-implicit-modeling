"""Edit waterfalls through the GRID probes — the picture behind `scores/summary*.md`.

Columns: GT, unsteered, and each editor at the arm the extended sweep reports (or a
fidelity-guarded arm), through the same rollout functions `edit_grid.py` scores with, on
the same case selection (teleport changes cell), drawn through `pim.figures.waterfall_grid`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import edit_grid as eg  # noqa: E402
from grid import G, cell_of, label_frames  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF  # noqa: E402
from pim.figures import waterfall_grid  # noqa: E402
from pim.metrics.edit_index import edit_index_per_case  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/noise_ablation/L-dw-noiseless-20m")
    ap.add_argument("--edit-json", default=str(eg.EXP / "scores/grid_edit_ext.json"))
    ap.add_argument("--probes-json", default=str(eg.EXP / "scores/grid_probes.json"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--guard", type=float, default=1.1)
    ap.add_argument("--n-ctx", type=int, default=6)
    ap.add_argument("--out", default=str(eg.EXP / "outputs/waterfall_grid_edits.png"))
    a = ap.parse_args()
    run_dir = REPO / a.run
    inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
    eval_dir = REPO / "datasets/discworld" / inst / "eval"
    model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    E = json.loads(Path(a.edit_json).read_text())
    P = json.loads(Path(a.probes_json).read_text())
    store = ProbeCache(eg.EXP / "probes")
    key = "best_guarded_1.1" if a.guard else "best"
    arms = {ed: E[key][ed] for ed in ("PI", "ND", "GS") if E[key].get(ed)}
    sel = np.array(E["select"][: a.n])
    b = dwb.load_bench(model, n=len(sel), target="full", basis_name="frustum", data_dir=eval_dir, select=sel)
    n, ar = b.n, np.arange(len(sel))
    sim = b.sim
    cur, _ = label_frames(b.pos[:, EF - 1], sim)
    j = b.edit_object.astype(int)
    A = cell_of(b.pos[ar, EF - 1, j], sim)
    B = cell_of(b.pos[ar, EF, j], sim)
    cls = j + 1
    tv = cur.astype(np.int64).copy(); tv[ar, A] = 0; tv[ar, B] = cls
    cm = np.zeros((n, G), bool); cm[ar, A] = True; cm[ar, B] = True
    A_t, B_t, cls_t = (torch.from_numpy(x).to(DEV) for x in (A, B, cls))
    cm_t, tv_t = torch.from_numpy(cm).to(DEV), torch.from_numpy(tv).to(DEV)
    lin = eg.load_probes(store, P, "linear", range(9))
    mlp = eg.load_probes(store, P, "mlp", range(9))
    rolls = {"unsteered": dwa.unsteered_rollout(model, b)}
    for ed, r in arms.items():
        pt, al = int(r["point"]), float(r["alpha"])
        if ed == "PI":
            roll = eg.pi_rollout(model, b, lin[pt][0], pt, al, A_t, B_t, cls_t, {})
        elif ed == "ND":
            roll = eg.nd_rollout(model, b, lin[pt][0], pt, al, A_t, B_t, cls_t)
        else:
            roll = eg.gs_rollout(model, b, mlp, pt, al, cm_t, tv_t, E["gs_steps"], E["beta"], {})
        rolls[f"{ed} (pt{pt} α{al:g})\n192-case EI {r['edit_index']:+.3f} / fid {r['fidelity_ratio']:.2f}"] = roll
    per = {nm: edit_index_per_case(r[:, 0], b.zones.gt_edited, b.zones.gt_unedited, b.zones.differing) for nm, r in rolls.items()}
    cx = lambda m: (np.where(m)[0].mean() if m.any() else np.nan)
    fig = waterfall_grid(
        columns=rolls, context=b.obs[:, EF - a.n_ctx: EF], gt=b.gt_roll,
        title=(f"{a.run} — editors through the GRID probes ({P['grid']['NU']}x{P['grid']['ND']} cells x 3), "
               f"first {n} cell-changing cases · EF={EF}, basis frustum, {inst}"),
        sample_idx=range(n),
        target_x=np.array([cx(b.zones.target[i]) for i in range(n)]),
        ghost_x=np.array([cx(b.zones.ghost[i]) for i in range(n)]),
        metrics={nm: float(np.nanmean(v)) for nm, v in per.items()}, metric_label="drawn rows: Edit Index")
    for r, i in enumerate(sel):
        fig.axes[r * (len(rolls) + 1)].set_ylabel(f"case {i} · obj {j[r]}\ncell {A[r]}→{B[r]}\ntime ↓", color="#c9d1e0", fontsize=7.5)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("arms:", {k: (v["point"], v["alpha"]) for k, v in arms.items()})
    print("per-case step-0 EI over the drawn rows:")
    for nm, v in per.items():
        print(f"  {nm.split(chr(10))[0]:22s}", " ".join(f"{x:+.2f}" for x in v))
    print("->", a.out)


if __name__ == "__main__":
    main()
