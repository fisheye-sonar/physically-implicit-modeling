"""dw-blink stage E: editability on the blink subsets + hidden-frame decodability (2026-09-07).

Subsets of the edits split at the canonical edit frame EF=20 (edited object's schedule):
  reappearance  hidden through EF-1, visible at EF (staleness k = hidden frames before EF);
                also the k >= 3 slice — the model must carry the position for >= 3 frames
  mid_blackout  hidden AT EF: the edit lands while the object is invisible; step 0 has no
                differing rays (NaN by construction), so the case is scored at the step
                the object reappears (`ei_at_reappearance`, per case, then averaged)
  visible       visible at EF-1 and EF: the within-instance control (same model, same
                probes, same editors, same grids)
Every subset gets the canonical PI and GS sweeps (all points / start layers, the
master_eval α grids, both dim sets) through the canonical arms, on the run's own cached
probes. ND is computed nowhere here — it is not reported for discworld (SETTINGS).

Then decodability by visibility: on probe-split sequences the fits never saw, the
canonical linear and MLP probes' skill on the position of an object at frames where it is
hidden vs visible, and as a function of frames since it was last seen.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, h5py

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint, n_points
from pim.environments.discworld import arms as dwa
from pim.environments.discworld import bench as dwb
from pim.environments.discworld.bench import EF, K_ROLL, N_OBJ, _to_basis
from pim.metrics.edit_index import edit_index_per_case
from pim.probes.base import collect_residuals
from pim.metrics.decodability import probe_skill_regression

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_PI = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0)
ALPHA_GS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/blink_ablation/L-dw-blink-20m")
ap.add_argument("--eval-dir", default=None, help="edits split dir (default: the run's instance eval/)")
ap.add_argument("--probe-corpus", default=None, help="probe split dir (default: instance probe/)")
ap.add_argument("--probe-cache", default=None, help="probe cache dir (default: the run's probes/)")
ap.add_argument("--probe-seqs", type=int, default=30_000)
ap.add_argument("--n", type=int, default=192, help="cases per subset")
ap.add_argument("--bases", nargs="+", default=["cartesian", "frustum"])
ap.add_argument("--out", default="experiments/blink_ablation/scores")
ap.add_argument("--quick", action="store_true", help="smoke: 2 alphas, 1 point, 1 start layer")
a = ap.parse_args()

run = REPO / a.run
cfg = json.loads((run / "config.json").read_text())
inst = cfg.get("data", {}).get("instance", "dw-blink")
inst_root = REPO / "datasets" / "discworld" / inst
eval_dir = Path(a.eval_dir) if a.eval_dir else inst_root / "eval"
probe_corpus = Path(a.probe_corpus) if a.probe_corpus else inst_root / "probe"
probe_cache = Path(a.probe_cache) if a.probe_cache else run / "probes"
out_dir = REPO / a.out
out_dir.mkdir(parents=True, exist_ok=True)
alpha_pi = ALPHA_PI[3:5] if a.quick else ALPHA_PI
alpha_gs = ALPHA_GS[3:5] if a.quick else ALPHA_GS

model, info = load_checkpoint(run / "best_model.pt", device=DEV)
NP = n_points(model)
points = [NP // 2] if a.quick else list(range(NP))
print(f"run {a.run}  arch {info.arch}  val {info.val_loss}  points {NP}  eval {eval_dir}", flush=True)

# ── subsets ───────────────────────────────────────────────────────────────────
with h5py.File(eval_dir / "edits.h5", "r") as f:
    ve = f["blink_visible"][:, :, :N_OBJ].astype(bool)
    eobj = f["edit_object"][:].astype(int)
    assert int(f["edit_frame"][0]) == EF
M = len(eobj); idx = np.arange(M)
vis_e = ve[idx, :, eobj]                                    # (M, T) the edited object
k = np.zeros(M, int)
for i in range(M):
    t = EF - 1
    while t >= 0 and not vis_e[i, t]:
        k[i] += 1; t -= 1
reapp_step = np.zeros(M, int)                               # first visible step >= EF
for i in np.where(~vis_e[:, EF])[0]:
    s = 0
    while EF + s < vis_e.shape[1] and not vis_e[i, EF + s]:
        s += 1
    reapp_step[i] = s
subsets = {
    "reappearance": np.where((k >= 1) & vis_e[:, EF])[0],
    "reappearance_k3": np.where((k >= 3) & vis_e[:, EF])[0],
    "mid_blackout": np.where(~vis_e[:, EF])[0],
    "visible": np.where(vis_e[:, EF - 1] & vis_e[:, EF])[0],
}
counts = {s: int(len(v)) for s, v in subsets.items()}
subsets = {s: v[: a.n] for s, v in subsets.items()}
print("subset sizes (available -> used):", {s: (counts[s], len(v)) for s, v in subsets.items()}, flush=True)
if a.quick:
    subsets = {s: v[:24] for s, v in subsets.items()}

def ei_at_reappearance(roll, b, sel):
    """Per-case Edit Index at each case's own reappearance step (0 for a visible object)."""
    z = b.zones
    r = reapp_step[sel]
    r = np.minimum(r, roll.shape[1] - 1)
    ii = np.arange(len(sel))
    ei = edit_index_per_case(roll[ii, r], b.gt_roll[ii, r], z.gt_unedited_traj[ii, r],
                             z.differing_traj[ii, r])
    return float(np.nanmean(ei)), int(np.isfinite(ei).sum())

def card(model, b, roll, u_card, sel):
    c = dwa.score(model, b, roll, u_card)
    c["ei_at_reappearance"], c["n_scored_at_reappearance"] = ei_at_reappearance(roll, b, sel)
    return {k_: v for k_, v in c.items() if np.isscalar(v) or k_ == "edit_index_by_step"}

results = {"run": a.run, "instance": inst, "eval_dir": str(eval_dir), "n_per_subset": a.n,
           "available": counts, "staleness_hist": {int(kk): int(c) for kk, c in
                                                   zip(*np.unique(k[k > 0], return_counts=True))},
           "bases": {}}
for basis in a.bases:
    t0 = time.time()
    lin = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="linear",
                         basis_name=basis, data_dir=probe_corpus, cache_dir=probe_cache, log=print)
    mlp = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="mlp",
                         basis_name=basis, data_dir=probe_corpus, cache_dir=probe_cache, log=print)
    B = {"probe_skill_linear": {int(e): float(s["r2"]) for e, (p, s) in lin.items()},
         "probe_skill_mlp": {int(e): float(s["r2"]) for e, (p, s) in mlp.items()}, "subsets": {}}
    for name, sel in subsets.items():
        if len(sel) == 0:
            print(f"  [{basis}] {name}: no cases", flush=True); continue
        b = dwb.load_bench(model, n=len(sel), target="full", basis_name=basis,
                           data_dir=eval_dir, select=sel)
        u_roll = dwa.unsteered_rollout(model, b)
        u = dwa.score(model, b, u_roll)
        u["fidelity_ratio"] = 1.0
        u["ei_at_reappearance"], u["n_scored_at_reappearance"] = ei_at_reappearance(u_roll, b, sel)
        arms = []
        for dims in ("pos", "all"):
            for ell in points:
                for al in alpha_pi:
                    roll = dwa.pinv_rollout(model, b, lin[ell][0], ell, al, space="zspace", dims=dims)
                    arms.append({"editor": "PI", "point": ell, "alpha": al, "dims": dims,
                                 **card(model, b, roll, u, sel)})
            for ls in ([points[0]] if a.quick else range(NP)):
                for al in alpha_gs:
                    roll = dwa.grad_steer_rollout(model, b, mlp, ls, al, n_steps=100, beta=0.2, dims=dims)
                    arms.append({"editor": "GS", "point": ls, "alpha": al, "dims": dims,
                                 **card(model, b, roll, u, sel)})
        key = "ei_at_reappearance" if name == "mid_blackout" else "edit_index"
        best = {}
        for ed in ("PI", "GS"):
            recs = [r for r in arms if r["editor"] == ed and np.isfinite(r[key])]
            best[ed] = max(recs, key=lambda r: r[key]) if recs else None
        B["subsets"][name] = {"n": int(len(sel)), "select": sel.tolist(),
                              "staleness_mean": float(k[sel].mean()),
                              "unedited": {k_: v for k_, v in u.items() if np.isscalar(v)},
                              "best": best, "arms": arms}
        line = " ".join(f"{ed} {v[key]:+.3f}/fid {v['fidelity_ratio']:.2f} (pt {v['point']} α {v['alpha']} {v['dims']})"
                        for ed, v in best.items() if v)
        print(f"  [{basis}] {name:16s} n={len(sel):3d}  unedited {u[key]:+.3f}  {line}", flush=True)
    results["bases"][basis] = B
    (out_dir / "subset_editability.json").write_text(json.dumps(results, indent=1))
    print(f"  [{basis}] done in {(time.time() - t0) / 60:.1f} min", flush=True)

# ── decodability by visibility (probe-split sequences the fits never saw) ──────
dec = {}
n_seq = 4000 if not a.quick else 200
start = a.probe_seqs                                        # fits used [0, probe_seqs)
with h5py.File(probe_corpus / "test.h5", "r") as f:
    if f["obs_intensity"].shape[0] < start + n_seq:
        start = max(0, f["obs_intensity"].shape[0] - n_seq)
    obs = f["obs_intensity"][start:start + n_seq].astype(np.float32)
    pos = f["positions"][start:start + n_seq, :, :N_OBJ, :].astype(np.float32)
    vel = f["velocities"][start:start + n_seq, :, :N_OBJ, :].astype(np.float32)
    vis = f["blink_visible"][start:start + n_seq, :, :N_OBJ].astype(bool)
sim = json.load(open(probe_corpus / "dataset.json"))["sim"]
span = getattr(model, "state_span", obs.shape[1])
obs = obs[:, :span]
since = np.zeros(vis.shape, int)                            # frames since last seen (0 = visible)
for t in range(1, vis.shape[1]):
    since[:, t] = np.where(vis[:, t], 0, since[:, t - 1] + 1)
for basis in a.bases:
    lin = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="linear",
                         basis_name=basis, data_dir=probe_corpus, cache_dir=probe_cache, log=None)
    mlp = dwa.fit_probes(model, target="full", n_seq=a.probe_seqs, family="mlp",
                         basis_name=basis, data_dir=probe_corpus, cache_dir=probe_cache, log=None)
    bp, bv = _to_basis(pos, vel, sim, basis)
    y = np.concatenate([bp.reshape(n_seq, bp.shape[1], -1), bv.reshape(n_seq, bv.shape[1], -1)], -1)
    y = y[:, :span]
    dec[basis] = {}
    for fam, probes in (("linear", lin), ("mlp", mlp)):
        ell = max(probes, key=lambda e: probes[e][1]["r2"])
        R = collect_residuals(model, obs, batch=64, points=[ell])[0]          # (N, T, d)
        T = R.shape[1]
        with torch.no_grad():
            pred = np.concatenate([probes[ell][0](torch.from_numpy(R[i:i + 256]).to(DEV)
                                                  .reshape(-1, R.shape[-1])).cpu().numpy()
                                   .reshape(-1, T, y.shape[-1]) for i in range(0, n_seq, 256)])
        yt = y[:, :T]; v = vis[:, :T]; sc = since[:, :T]
        row = {"point": int(ell)}
        for j in range(N_OBJ):
            d = [2 * j, 2 * j + 1]                          # object j's position read-outs
            yy, pp = yt[:, :, d], pred[:, :, d]
            mean = yy.reshape(-1, 2).mean(0)
            def skill(mask):
                if mask.sum() < 50:
                    return None
                return float(probe_skill_regression(pp[mask], yy[mask], mean))
            row[f"obj{j}"] = {
                "visible": skill(v[:, :, j]), "hidden": skill(~v[:, :, j]),
                "by_frames_since_seen": {int(s): skill(sc[:, :, j] == s) for s in range(0, 13)},
                "n_hidden_frames": int((~v[:, :, j]).sum()),
            }
        dec[basis][fam] = row
        print(f"  decodability [{basis}/{fam}] point {ell}: " + "  ".join(
            f"obj{j} visible {row[f'obj{j}']['visible']:+.3f} hidden {row[f'obj{j}']['hidden']:+.3f}"
            for j in range(N_OBJ)), flush=True)
        del R
results["decodability_by_visibility"] = dec
(out_dir / "subset_editability.json").write_text(json.dumps(results, indent=1))

# ── summary ───────────────────────────────────────────────────────────────────
lines = [f"# dw-blink subset editability — {a.run}", "",
         f"cases available: {counts}; used up to {a.n} per subset; staleness hist {results['staleness_hist']}", "",
         "| basis | subset | n | unedited | PI best | fid | GS best | fid |", "|---|---|---|---|---|---|---|---|"]
for basis, B in results["bases"].items():
    for name, S in B["subsets"].items():
        key = "ei_at_reappearance" if name == "mid_blackout" else "edit_index"
        pi, gs = S["best"]["PI"], S["best"]["GS"]
        f = lambda r: (f"{r[key]:+.3f} (pt {r['point']}, α {r['alpha']}, {r['dims']})", f"{r['fidelity_ratio']:.2f}") if r else ("—", "—")
        lines.append(f"| {basis} | {name} | {S['n']} | {S['unedited'][key]:+.3f} | {f(pi)[0]} | {f(pi)[1]} | {f(gs)[0]} | {f(gs)[1]} |")
lines += ["", "mid_blackout is scored at each case's reappearance step (NaN at step 0 by construction).", "",
          "## decodability by visibility (held-out probe sequences, best point)", "",
          "| basis | probe | obj | visible | hidden | since 1 | 3 | 6 | 10 |", "|---|---|---|---|---|---|---|---|---|"]
for basis, D in dec.items():
    for fam, row in D.items():
        for j in range(N_OBJ):
            r = row[f"obj{j}"]; g = lambda s: ("—" if r["by_frames_since_seen"].get(s) is None else f"{r['by_frames_since_seen'][s]:+.3f}")
            fmt = lambda x: "—" if x is None else f"{x:+.3f}"
            lines.append(f"| {basis} | {fam} | {j} | {fmt(r['visible'])} | {fmt(r['hidden'])} | {g(1)} | {g(3)} | {g(6)} | {g(10)} |")
(out_dir / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
