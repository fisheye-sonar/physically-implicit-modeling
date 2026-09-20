"""History rewrite with the inverse map (2026-09-16, Sevan's idea).

The post-edit target state carries the edited object's velocity, so its counterfactual
trajectory can be integrated BACKWARDS over the whole teacher-forced history (constant
velocity, open boundary, no noise on dw-noiseless). At every history step t the model's
residual at the IM point is overwritten with g(s_cf[t]) — the inverse map of the
counterfactual full state — and the model's own prediction becomes the rewritten frame
t+1. The rewritten history then replaces the window the edit is launched from.

Frame 0 is kept from the original sequence (nothing precedes it to generate it from).
Arms (all scored on the canonical bench, first 32 selected cases):
  unsteered · IM (canonical: original history, g(s_post) at EF−1)
  hist+IM  (rewritten history, then the same write at EF−1)
  hist     (rewritten history, no write at EF−1 — does the history alone carry the edit?)
Also reported: RMSE of the rewritten history against the SIM'S clean counterfactual render.
Outputs land beside this script; everything canonical comes from pim.* — nothing under experiments/
(moved here 2026-09-19 from experiments/history_rewrite/, so the paper does not depend on that tree)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import h5py, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint
from pim.models.protocol import free_run
from pim.environments import layout
from pim.environments.discworld import arms as dwa, bench as dwb
from pim.environments.discworld.bench import full_state_pair, EF, K_ROLL, N_OBJ
from pim.environments.discworld.config import SimConfig
from pim.environments.discworld.renderer import render_frame
from pim.editors.inverse import inverse_overwrite
from pim.figures import waterfall_grid

RUN, BLOCK, N_BENCH, N_ROWS, N_CTX = "noise_ablation/L-dw-noiseless-20m", "cartesian", 32, 4, 6
OUT = Path(__file__).resolve().parent            # outputs land beside this script (paper/figs/history_rewrite/)
run_dir = REPO / "runs" / RUN
inst = json.loads((run_dir / "config.json").read_text())["data"]["instance"]
best = json.loads((run_dir / "scores.json").read_text())["bases"][BLOCK]["best"]["IM"]
PT = int(best["point"])
DEV = dwb.DEV
model, _ = load_checkpoint(run_dir / "best_model.pt", device=DEV)
b = dwb.load_bench(model, n=N_BENCH, target="full", basis_name=BLOCK, instance=inst)
recipe = dwa.probe_recipe("full", inst, n_seq=30_000)

# ── the counterfactual history: edited object integrated backwards from the pre-dynamics target
dt = float(b.sim["dt"]); ar = np.arange(b.n); eo = b.edit_object
pos_cf, vel_cf = b.pos[:, :EF].copy(), b.vel[:, :EF].copy()
pre_dyn = b.pos[ar, EF, eo] - b.vel[ar, EF - 1, eo] * dt               # the write target's position
v = b.vel[ar, EF - 1, eo]
for t in range(EF):
    pos_cf[ar, t, eo] = pre_dyn - (EF - 1 - t) * v * dt
    vel_cf[ar, t, eo] = v
s_cf = np.concatenate([pos_cf.reshape(b.n, EF, -1), vel_cf.reshape(b.n, EF, -1)], -1).astype(np.float32)
_, s_post = full_state_pair(b.pos, b.vel, b.edit_object, b.sim, BLOCK)
assert np.allclose(s_cf[:, EF - 1], s_post, atol=1e-4), "s_cf[EF-1] must equal the canonical write target"

# the sim's clean render of that counterfactual history (radii / reflectivities from the edits split)
sel = np.asarray(json.loads(layout.edits_selection("discworld", inst).read_text())["select"], int)[:N_BENCH] \
    if hasattr(layout, "edits_selection") else None
with h5py.File(layout.edits_file("discworld", inst), "r") as f:
    if sel is None:
        sp = Path(str(layout.edits_file("discworld", inst))).with_name("selection.json")
        sel = np.asarray(json.loads(sp.read_text())["select"], int)[:N_BENCH]
    radii, refl = f["radii"][:][sel, :N_OBJ], f["reflectivities"][:][sel, :N_OBJ]
    cfg = SimConfig(**json.loads(f.attrs["config_json"])["dataset"]["sim"])
cf_clean = np.stack([np.stack([render_frame(pos_cf[i, t], radii[i], refl[i], cfg)[2] for t in range(EF)])
                     for i in range(b.n)]).astype(np.float32)

# ── g at the IM point, then the per-step rewrite
g = next(gg for _, gg, _, _ in dwa.iter_inverse_maps(model, basis_name=BLOCK, points=[PT],
                                                     cache_dir=run_dir / "probes", log=None, **recipe))
S = torch.from_numpy(s_cf).to(DEV)
obs_cf = torch.from_numpy(b.obs[:, :EF]).to(DEV).clone()
with torch.no_grad():
    for t in range(EF - 1):
        st = model.state_from_obs(obs_cf[:, : t + 1])
        obs_cf[:, t + 1] = model.decode_with_edit(st, PT, inverse_overwrite(g, S[:, t]))
    h_post = inverse_overwrite(g, torch.from_numpy(s_post).to(DEV))
    st_cf = model.state_from_obs(obs_cf)
    rolls = {"unsteered": dwa.unsteered_rollout(model, b),
             "IM": model.rollout_with_edit(b.state, PT, h_post, K_ROLL).cpu().numpy(),
             "hist+IM": model.rollout_with_edit(st_cf, PT, h_post, K_ROLL).cpu().numpy(),
             "hist": free_run(model, model.decode(st_cf), model.advance(st_cf, model.decode(st_cf)), K_ROLL).cpu().numpy()}
obs_cf = obs_cf.cpu().numpy()
cards = {k: dwa.score(model, b, r) for k, r in rolls.items()}
uns = cards["unsteered"]
for k in cards: cards[k]["fidelity_ratio"] = 1.0 if k == "unsteered" else dwa.fidelity_ratio(cards[k], uns)
hist_rmse = {"original obs vs cf render": float(np.sqrt(((b.obs[:, 1:EF] - cf_clean[:, 1:]) ** 2).mean())),
             "rewritten vs cf render": float(np.sqrt(((obs_cf[:, 1:] - cf_clean[:, 1:]) ** 2).mean())),
             "rewritten vs original obs": float(np.sqrt(((obs_cf[:, 1:] - b.obs[:, 1:EF]) ** 2).mean()))}

print(f"{RUN} · IM point {PT} · {b.n} cases\n{'arm':<10}{'EI':>8}{'fid':>7}{'target':>8}{'ghost':>8}{'collat':>8}")
for k, c in cards.items():
    print(f"{k:<10}{c['edit_index']:>+8.3f}{c['fidelity_ratio']:>7.2f}{c['target_rmse']:>8.4f}{c['ghost_rmse']:>8.4f}{c['collateral_rmse']:>8.4f}")
print("history RMSE:", {k: round(v, 4) for k, v in hist_rmse.items()})
json.dump({"run": RUN, "block": BLOCK, "point": PT, "n": b.n, "cards": cards, "history_rmse": hist_rmse},
          open(OUT / "scores.json", "w"), indent=1)

def _cx(m):
    i = np.where(m)[0]; return i.mean() if i.size else np.nan
tx = np.array([_cx(b.zones.target[i]) for i in range(N_ROWS)]); gx = np.array([_cx(b.zones.ghost[i]) for i in range(N_ROWS)])
R = range(N_ROWS)
lab = {k: f"{k} (EI {cards[k]['edit_index']:+.2f}, fid {cards[k]['fidelity_ratio']:.2f})" for k in rolls}
fig = waterfall_grid(columns={lab[k]: rolls[k][:N_ROWS] for k in ("unsteered", "IM")},
                     context=b.obs[:N_ROWS, EF - N_CTX:EF], gt=b.gt_roll[:N_ROWS], sample_idx=R, target_x=tx, ghost_x=gx,
                     metrics={lab[k]: cards[k]["edit_index"] for k in ("unsteered", "IM")},
                     title=f"{RUN} — ORIGINAL history: unsteered and canonical IM (pt {PT}) at EF={EF}")
fig.savefig(OUT / "A_original_history_arms.png", dpi=120, bbox_inches="tight"); plt.close(fig)
fig = waterfall_grid(columns={lab[k]: rolls[k][:N_ROWS] for k in ("hist", "hist+IM")},
                     context=obs_cf[:N_ROWS, EF - N_CTX:EF], gt=b.gt_roll[:N_ROWS], sample_idx=R, target_x=tx, ghost_x=gx,
                     metrics={lab[k]: cards[k]["edit_index"] for k in ("hist", "hist+IM")},
                     title=f"{RUN} — REWRITTEN history (IM at every step, pt {PT}): free-run, and IM at EF={EF} on top")
fig.savefig(OUT / "B_rewritten_history_arms.png", dpi=120, bbox_inches="tight"); plt.close(fig)
fig = waterfall_grid(columns={"original obs (frames 1..19)": b.obs[:N_ROWS, 1:EF],
                              "rewritten history (IM every step)": obs_cf[:N_ROWS, 1:]},
                     context=b.obs[:N_ROWS, :1], gt=cf_clean[:N_ROWS, 1:], sample_idx=R,
                     gt_label="sim clean render of the COUNTERFACTUAL history",
                     metrics={"original obs (frames 1..19)": hist_rmse["original obs vs cf render"],
                              "rewritten history (IM every step)": hist_rmse["rewritten vs cf render"]},
                     metric_label="RMSE vs cf render",
                     title=f"{RUN} — the history itself: frames 1..{EF-1} (row above the line = the kept frame 0)")
fig.savefig(OUT / "C_history_vs_counterfactual_render.png", dpi=120, bbox_inches="tight"); plt.close(fig)
print("→", OUT.relative_to(REPO))
