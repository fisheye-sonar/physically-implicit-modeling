"""Stage B — editability through the grid-cell probes: PI, ND and GS on the canonical bench.

Same trained model, same 192-case edit set (first 192 cases whose teleport CHANGES cell —
a same-cell teleport is a no-op under this target and is dropped), same ray-zone Edit
Index and fidelity guard as the canonical scoring. Only the probe target differs: the
state is a 128-cell × 3-class grid (`grid.py`), so every editor is Othello's form:

  PI   the linear probe's logits at the edited object's OLD cell A and NEW cell B are
       swapped between the object's class and "empty" (Othello's tile swap, on two cells),
       and the residual is re-solved in z-space (`pim.editors.pinv.inject_state`) —
       `othello.arms.linear_arm`'s pinv branch, at every point, alpha swept.
  ND   `pim.editors.nanda.probe_direction(per_sample=True)`: the probe row of (B, class)
       minus the row of (A, class) — the "move the object" direction — added at alpha times
       the activation norm, every point, alpha swept. ND is APPLICABLE here (a categorical
       edit, as in Othello), unlike on the regression target.
  GS   `pim.editors.grad_steer` with the MLP probes' cross-entropy (Li's own loss): change
       mask on cells A and B, target labels {A: empty, B: object}, hold-the-rest beta 0.2,
       100 Adam steps, every start layer, alpha swept — the canonical discworld grid.

Rollouts go through the canonical `pim.environments.discworld.arms` helpers and the
scorecard through `arms.score`, so every number is on the axis of Table 2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import G, N_CLASSES, N_OBJ, TAG, cell_of, label_frames  # noqa: E402

from pim.editors.grad_steer import build_edit_spec, make_intervention_hook  # noqa: E402
from pim.editors.nanda import addition_hook, probe_direction  # noqa: E402
from pim.editors.pinv import inject_state  # noqa: E402
from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.bench import EF, K_ROLL  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "grid_target_control"
DEV = "cuda"
ALPHA_PI = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
ALPHA_ND = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
ALPHA_GS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5)


def load_probes(store, results, fam, points):
    out = {}
    for ell in points:
        c = results["points"][str(ell)][fam]["cache"]
        hit = store.load(c["file"], c["prov"], device=DEV)
        if hit is None:
            raise FileNotFoundError(f"probe {c['file']} missing for point {ell} {fam}")
        out[ell] = hit
    return out


@torch.no_grad()
def pi_rollout(model, b, probe, ell, alpha, A, B, cls, rec):
    """Swap (empty, cls) logits at cells A and B, re-solve in z-space, roll out."""
    dwa.as_activations(model, ell)
    h0 = model.flat_state(b.state)
    n = h0.shape[0]
    ar = torch.arange(n, device=DEV)
    W = probe.net.weight.detach()
    Wp, bias = torch.linalg.pinv(W), probe.net.bias.detach()
    z = (h0 - probe.x_mean) / probe.x_std
    lg = probe.net(z).view(n, G, N_CLASSES).clone()
    for cell in (A, B):
        sel = lg[ar, cell].clone()
        e, c = sel[ar, 0].clone(), sel[ar, cls].clone()
        sel[ar, 0], sel[ar, cls] = c, e
        lg[ar, cell] = sel
    z_new = inject_state(z, lg.view(n, -1), W, Wp, bias)
    delta = alpha * (z_new - z) * probe.x_std
    h = h0 + delta
    rec["write_ratio"] = float((delta.norm(dim=1) / h0.norm(dim=1)).mean())
    lab = probe(h).argmax(-1)
    rec["readout_landed"] = float(((lab[ar, A] == 0) & (lab[ar, B] == cls)).float().mean())
    return model.rollout_with_edit(b.state, ell, h, K_ROLL).cpu().numpy()


@torch.no_grad()
def nd_rollout(model, b, probe, ell, alpha, A, B, cls):
    rows = B * N_CLASSES + cls
    sub = A * N_CLASSES + cls
    d = probe_direction(probe, rows, subtract_rows=sub, per_sample=True)
    return dwa._roll_hook(model, b.state, addition_hook(ell, d, alpha))


def gs_rollout(model, b, mlp, start, alpha, cm_t, tv_t, n_steps, beta, rec):
    pts = {e: mlp[e][0] for e in mlp if e >= start}
    specs = {}
    for e, pr in pts.items():
        dwa.as_activations(model, e)
        with torch.no_grad():
            x0 = model.flat_state(b.state)
        specs[e] = build_edit_spec(pr, x0, cm_t, tv_t, beta=beta)
    hook = make_intervention_hook(pts, specs, start, alpha=alpha, n_steps=n_steps, record=rec)
    return dwa._roll_hook(model, b.state, hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/noise_ablation/L-dw-noiseless-20m")
    ap.add_argument("--probes-json", default=str(EXP / f"scores/grid_probes{TAG}.json"))
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--gs-steps", type=int, default=100)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--points", type=int, nargs="*", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--alpha-pi", type=float, nargs="+", default=None)
    ap.add_argument("--alpha-nd", type=float, nargs="+", default=None)
    ap.add_argument("--alpha-gs", type=float, nargs="+", default=None)
    ap.add_argument("--out", default=str(EXP / f"scores/grid_edit{TAG}.json"))
    a = ap.parse_args()
    t0 = time.time()
    run_dir = REPO / a.run
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = cfg["data"]["instance"]
    eval_dir = REPO / "datasets/discworld" / inst / "eval"
    model, info = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    NP = n_points(model)
    res_p = json.loads(Path(a.probes_json).read_text())
    points = a.points if a.points else [e for e in range(NP) if str(e) in res_p["points"]]
    alpha_pi, alpha_nd, alpha_gs = ALPHA_PI, ALPHA_ND, ALPHA_GS
    if a.quick:
        alpha_pi, alpha_nd, alpha_gs = (0.5, 1.0), (0.2, 0.5), (0.02, 0.05)
    alpha_pi = tuple(a.alpha_pi) if a.alpha_pi else alpha_pi
    alpha_nd = tuple(a.alpha_nd) if a.alpha_nd else alpha_nd
    alpha_gs = tuple(a.alpha_gs) if a.alpha_gs else alpha_gs
    store = ProbeCache(EXP / "probes")
    lin = load_probes(store, res_p, "linear", points)
    mlp = load_probes(store, res_p, "mlp", points)

    # the bench: the first n cases whose teleport changes cell
    arr = dwb.bench_arrays(n=min(2000, 4 * a.n + 200), target="full", basis_name="frustum",
                           data_dir=eval_dir)
    sim = arr["sim"]
    eobj = arr["edit_object"]
    idx = np.arange(len(eobj))
    A_all = cell_of(arr["pos"][idx, EF - 1, eobj], sim)
    B_all = cell_of(arr["pos"][idx, EF, eobj], sim)
    valid = np.where(A_all != B_all)[0]
    sel = valid[: a.n]
    n_same = int(len(idx) - len(valid))
    b = dwb.load_bench(model, n=len(sel), target="full", basis_name="frustum",
                       data_dir=eval_dir, select=sel)
    n = b.n
    ar = np.arange(n)
    cur, _ = label_frames(b.pos[:, EF - 1], sim)                     # (n, G) at the model's current frame
    j = b.edit_object.astype(int)
    A = cell_of(b.pos[ar, EF - 1, j], sim)
    B = cell_of(b.pos[ar, EF, j], sim)
    cls = j + 1
    tv = cur.astype(np.int64).copy()
    tv[ar, A] = 0
    tv[ar, B] = cls
    cm = np.zeros((n, G), bool)
    cm[ar, A] = True
    cm[ar, B] = True
    A_t, B_t, cls_t = (torch.from_numpy(x).to(DEV) for x in (A, B, cls))
    cm_t, tv_t = torch.from_numpy(cm).to(DEV), torch.from_numpy(tv).to(DEV)
    print(f"{a.run} · {inst} · {n} cases (dropped {n_same} same-cell teleports of the first {len(idx)}) · "
          f"points {points} · probes from {Path(a.probes_json).name}", flush=True)

    u = dwa.unsteered(model, b)
    arms = []
    for ell in points:
        for al in alpha_pi:
            rec = {}
            roll = pi_rollout(model, b, lin[ell][0], ell, al, A_t, B_t, cls_t, rec)
            arms.append({"editor": "PI", "point": ell, "alpha": al, **rec,
                         **{k: v for k, v in dwa.score(model, b, roll, u).items()
                            if np.isscalar(v) or k == "edit_index_by_step"}})
        for al in alpha_nd:
            roll = nd_rollout(model, b, lin[ell][0], ell, al, A_t, B_t, cls_t)
            arms.append({"editor": "ND", "point": ell, "alpha": al, "write_ratio": float(al),
                         **{k: v for k, v in dwa.score(model, b, roll, u).items()
                            if np.isscalar(v) or k == "edit_index_by_step"}})
        best_pi = max((r for r in arms if r["editor"] == "PI" and r["point"] == ell), key=lambda r: r["edit_index"])
        best_nd = max((r for r in arms if r["editor"] == "ND" and r["point"] == ell), key=lambda r: r["edit_index"])
        print(f"  point {ell}: PI best {best_pi['edit_index']:+.3f}/fid {best_pi['fidelity_ratio']:.2f} (α {best_pi['alpha']}, "
              f"landed {best_pi['readout_landed']:.2f}) · ND best {best_nd['edit_index']:+.3f}/fid {best_nd['fidelity_ratio']:.2f} "
              f"(α {best_nd['alpha']})  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    for ls in points:
        for al in alpha_gs:
            rec = {}
            roll = gs_rollout(model, b, mlp, ls, al, cm_t, tv_t, a.gs_steps, a.beta, rec)
            wr = float(np.mean([d["delta_norm"] / d["x_norm"] for d in rec.values()
                                if isinstance(d, dict) and d.get("x_norm", 0) > 0] or [np.nan]))
            arms.append({"editor": "GS", "point": ls, "alpha": al, "write_ratio": wr,
                         **{k: v for k, v in dwa.score(model, b, roll, u).items()
                            if np.isscalar(v) or k == "edit_index_by_step"}})
        best_gs = max((r for r in arms if r["editor"] == "GS" and r["point"] == ls), key=lambda r: r["edit_index"])
        print(f"  GS from {ls}: best {best_gs['edit_index']:+.3f}/fid {best_gs['fidelity_ratio']:.2f} (α {best_gs['alpha']})"
              f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)

    def best(ed, guard=None):
        rs = [r for r in arms if r["editor"] == ed and (guard is None or r["fidelity_ratio"] <= guard)]
        return max(rs, key=lambda r: r["edit_index"]) if rs else None

    out = {"run": a.run, "instance": inst, "n_cases": int(n), "dropped_same_cell": n_same,
           "alphas": {"PI": list(alpha_pi), "ND": list(alpha_nd), "GS": list(alpha_gs)},
           "select": sel.tolist(), "points": points, "gs_steps": a.gs_steps, "beta": a.beta,
           "grid": res_p["grid"], "probe_n_seq": res_p["n_seq"], "probe_epochs": res_p["epochs"],
           "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
           "best": {ed: best(ed) for ed in ("PI", "ND", "GS")},
           "best_guarded_1.1": {ed: best(ed, 1.1) for ed in ("PI", "ND", "GS")},
           "arms": arms, "minutes": (time.time() - t0) / 60}
    # the canonical regression-target numbers of the same run, for the table
    sp = run_dir / "scores.json"
    if sp.exists():
        s = json.loads(sp.read_text())["bases"]["frustum"]
        out["canonical_frustum"] = {"unedited": s["unedited"]["edit_index"],
                                    "best": {k: {kk: vv for kk, vv in v.items() if np.isscalar(vv)}
                                             for k, v in s["best"].items() if v},
                                    "probe_skill_linear": max(s["probe_skill_linear"]),
                                    "probe_skill_mlp": max(s["probe_skill_mlp"])}
    Path(a.out).write_text(json.dumps(out, indent=1))

    lines = [f"# grid-target control — {a.run} ({inst})", "",
             f"grid {res_p['grid']['NU']}x{res_p['grid']['ND']} = {G} cells x {N_CLASSES} classes; probes on "
             f"{res_p['n_seq']:,} sequences, {res_p['epochs']} epochs; bench {n} cases "
             f"({n_same} same-cell teleports dropped); GS {a.gs_steps} steps, beta {a.beta}", "",
             "| target | probe skill LIN / MLP | unedited | PI | ND | GS |", "|---|---|---|---|---|---|"]
    f = lambda r: "—" if r is None else f"{r['edit_index']:+.3f} / fid {r['fidelity_ratio']:.2f} (pt {r['point']}, α {r['alpha']:g})"
    bp = res_p.get("best_point", {})
    sk = lambda fam: res_p["points"][bp[fam]][fam]["skill"] if fam in bp else float("nan")
    lines.append(f"| grid 3-way (this) | {sk('linear'):+.3f} / {sk('mlp'):+.3f} | {u['edit_index']:+.3f} | "
                 f"{f(out['best']['PI'])} | {f(out['best']['ND'])} | {f(out['best']['GS'])} |")
    lines.append(f"| grid, fidelity ≤ 1.1 | | | {f(out['best_guarded_1.1']['PI'])} | "
                 f"{f(out['best_guarded_1.1']['ND'])} | {f(out['best_guarded_1.1']['GS'])} |")
    if "canonical_frustum" in out:
        c = out["canonical_frustum"]
        g = lambda k: f(c["best"].get(k))
        lines.append(f"| regression, frustum (canonical scores.json) | {c['probe_skill_linear']:+.3f} / {c['probe_skill_mlp']:+.3f} | "
                     f"{c['unedited']:+.3f} | {g('PI')} | {g('ND')} (not reported for discworld) | {g('GS')} |")
    lines += ["", "Probe skill for the grid target is 1 − err / err(majority) (classification, majority = empty); "
              "for the regression target it is R² — same axis, different formula. ND on the canonical row is "
              "computed but not reported (a fixed direction is incoherent for continuous teleports); on the grid "
              "target it is a categorical edit and IS reported."]
    (Path(a.out).with_name(Path(a.out).stem.replace("grid_edit", "summary") + ".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
