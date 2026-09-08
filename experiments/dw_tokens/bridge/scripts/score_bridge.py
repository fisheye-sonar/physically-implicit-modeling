#!/usr/bin/env python
"""Score the frames-as-tokens run with the ORIGINAL discworld analysis, through the adapter.

Everything is the canonical discworld bench — `load_bench` (192 teleports, EF=20, K=15),
`unsteered`, `pinv_arm` (PI z-space), `grad_steer_arm` (GS), `edit_scorecard` (ray-zone Edit
Index, zone RMSEs), `fidelity_ratio` — driven through `TokenFrameAdapter`, which renders the
next-frame distribution as the EXPECTED frame and feeds the ARGMAX token back in rollouts.
Probes are the run's own cached ones (fitted by master_eval on token inputs; a cache miss
aborts — nothing is refitted here). Settings mirror master_eval's SETTINGS.

A second, smaller pass renders the ARGMAX frame instead (unedited + each editor's best arm),
so the two renderings can be compared on the same arms.

Output: scores/bridge_<run>.json (the discworld block schema + "argmax_rendering") and
scores/summary.md beside the regression run's numbers and the frame-set numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import TokenFrameAdapter  # noqa: E402

from pim.environments.discworld import arms as dwa  # noqa: E402
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.environments.discworld.token_bench import token_encoder  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab  # noqa: E402
from pim.models import load_checkpoint, n_points  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

EXP = REPO / "experiments" / "dw_tokens" / "bridge"
DEV = dwb.DEV
SETTINGS = {
    "dw_probe_seqs": 30_000, "dw_bench_n": 192, "dw_target": "full", "dw_edit_dims": ("pos", "all"),
    "dw_bases": ("cartesian", "frustum"),
    "dw_alpha_pi": (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, 175.0),
    "dw_alpha_gs": (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5),
    "dw_gs_steps": 100, "dw_gs_beta": 0.2,
}


def cached_probes(model, run_dir: Path, inst: Path, basis: str, family: str, n_seq: int, vocab):
    """The run's own probes, by the exact key master_eval used (model fingerprint + encoder tag)."""
    store = ProbeCache(run_dir / "probes")
    _, tag = token_encoder(vocab)
    fname, prov = store.key(model, target="full", n_seq=int(n_seq), split="test", family=family,
                            basis=basis, seed=dwb.SEED, data=str((inst / "probe").resolve()), encoder=tag)
    hit = store.load(fname, prov, device=DEV)
    if hit is None:
        sys.exit(f"probe cache MISS for {family}/{basis} under {run_dir / 'probes'} — refusing to refit")
    return hit


def best_arm(recs):
    b = max(recs, key=lambda r: r["edit_index"])
    return {k: v for k, v in b.items() if np.isscalar(v)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="runs/interface_ablation/L-dw-8ray-tok-20m")
    ap.add_argument("--smoke", action="store_true", help="tiny grids on the smoke run's 400-seq probes")
    a = ap.parse_args()
    s = dict(SETTINGS)
    if a.smoke:
        s.update({"dw_probe_seqs": 400, "dw_bench_n": 24, "dw_alpha_pi": (1.0, 20.0, 175.0),
                  "dw_alpha_gs": (0.05, 0.5), "dw_gs_steps": 5, "dw_bases": ("frustum",)})
    t0 = time.time()
    run_dir = (REPO / a.run).resolve()
    cfg = json.loads((run_dir / "config.json").read_text())
    inst = REPO / "datasets" / "discworld" / cfg["data"]["instance"]
    model, info = load_checkpoint(run_dir / "best_model.pt", device=DEV)
    model.eval()
    vocab = FrameVocab.load(run_dir / "vocab.npz")
    ad = TokenFrameAdapter(model, vocab, render="expected", feedback="argmax").eval()
    out = {"run": run_dir.name, "arch": info.arch, "val_loss": info.val_loss, "render": "expected",
           "feedback": "argmax", "ei_construction": "ray-zone (bridge)", "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
           "bases": {}}
    for basis in s["dw_bases"]:
        b = dwb.load_bench(ad, n=s["dw_bench_n"], target=s["dw_target"], basis_name=basis, data_dir=inst / "eval")
        lin = cached_probes(model, run_dir, inst, basis, "linear", s["dw_probe_seqs"], vocab)
        mlp = cached_probes(model, run_dir, inst, basis, "mlp", s["dw_probe_seqs"], vocab)
        ad.render = "expected"
        u = dwa.unsteered(ad, b)
        arms = []
        for dims in s["dw_edit_dims"]:
            arms += dwa.pinv_arm(ad, b, lin, s["dw_alpha_pi"], space="zspace", dims=dims)
            arms += dwa.grad_steer_arm(ad, b, mlp, range(n_points(model)), s["dw_alpha_gs"],
                                       n_steps=s["dw_gs_steps"], beta=s["dw_gs_beta"], dims=dims)
        for r in arms:
            r["fidelity_ratio"] = dwa.fidelity_ratio(r, u)

        def pick(ed, dims=None):
            sub = [r for r in arms if r["editor"].startswith(ed) and (dims is None or r["dims"] == dims)]
            return best_arm(sub) if sub else None
        block = {"unedited": {k: v for k, v in u.items() if np.isscalar(v)},
                 "probe_skill_linear": [v[1]["r2"] for v in lin.values()],
                 "probe_skill_mlp": [v[1]["r2"] for v in mlp.values()],
                 "best": {ed: pick(ed) for ed in ("PI", "GS")},
                 "best_by_dims": {d: {ed: pick(ed, d) for ed in ("PI", "GS")} for d in s["dw_edit_dims"]},
                 "arms": [{k: v for k, v in r.items() if np.isscalar(v)} for r in arms]}
        # the ARGMAX rendering on the same best arms
        ad.render = "argmax"
        ua = dwa.unsteered(ad, b)
        arg = {"unedited": {k: v for k, v in ua.items() if np.isscalar(v)}}
        for ed, bb in block["best"].items():
            if not bb:
                continue
            if ed == "PI":
                roll = dwa.pinv_rollout(ad, b, lin[bb["point"]][0], bb["point"], bb["alpha"], space="zspace", dims=bb["dims"])
            else:
                roll = dwa.grad_steer_rollout(ad, b, mlp, bb["point"], bb["alpha"], n_steps=s["dw_gs_steps"],
                                              beta=s["dw_gs_beta"], dims=bb["dims"])
            c = dwa.score(ad, b, roll, ua)
            arg[ed] = {**{k: v for k, v in c.items() if np.isscalar(v)}, "point": bb["point"], "alpha": bb["alpha"], "dims": bb["dims"]}
        block["argmax_rendering"] = arg
        ad.render = "expected"
        out["bases"][basis] = block
        pi, gs = block["best"]["PI"], block["best"]["GS"]
        print(f"  {basis}: unedited {u['edit_index']:+.4f} | PI {pi['edit_index']:+.4f}/{pi['fidelity_ratio']:.2f} "
              f"({pi['dims']}·pt{pi['point']}·α{pi['alpha']:g}) | GS {gs['edit_index']:+.4f}/{gs['fidelity_ratio']:.2f} "
              f"({gs['dims']}·pt{gs['point']}·α{gs['alpha']:g}) | argmax: unedited {ua['edit_index']:+.4f} "
              f"PI {arg['PI']['edit_index']:+.4f} GS {arg['GS']['edit_index']:+.4f}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    out["unk_inputs"] = int(ad.unk_inputs)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    tag = "_smoke" if a.smoke else ""
    path = EXP / "scores" / f"bridge_{run_dir.name}{tag}.json"
    path.write_text(json.dumps(out, indent=1, default=float))

    # summary beside the regression run (ray-zone) and the frame-set scores
    md = [f"# Bridge: {run_dir.name} scored with the discworld analysis (expected-frame rendering, argmax feedback)", ""]
    reg = REPO / "runs" / "ray_ablation" / "L-dw-8ray-20m" / "scores.json"
    fs = run_dir / "scores.json"
    regS = json.load(open(reg)) if reg.exists() else None
    fsS = json.load(open(fs)) if fs.exists() else None
    md += ["| basis | reading | unedited EI | PI best · EI / fid | GS best · EI / fid |", "|---|---|---|---|---|"]
    for basis, T in out["bases"].items():
        pi, gs = T["best"]["PI"], T["best"]["GS"]
        md.append(f"| {basis} | token model · ray-zone via bridge (expected frame) | {T['unedited']['edit_index']:+.3f} | "
                  f"{pi['dims']}·pt{pi['point']}·α{pi['alpha']:g} · {pi['edit_index']:+.3f} / {pi['fidelity_ratio']:.2f} | "
                  f"{gs['dims']}·pt{gs['point']}·α{gs['alpha']:g} · {gs['edit_index']:+.3f} / {gs['fidelity_ratio']:.2f} |")
        ar = T["argmax_rendering"]
        md.append(f"| {basis} | token model · ray-zone via bridge (argmax frame, same arms) | {ar['unedited']['edit_index']:+.3f} | "
                  f"{ar['PI']['edit_index']:+.3f} / {ar['PI']['fidelity_ratio']:.2f} | {ar['GS']['edit_index']:+.3f} / {ar['GS']['fidelity_ratio']:.2f} |")
        if fsS and basis in fsS["bases"]:
            F = fsS["bases"][basis]
            fp, fg = F["best"]["PI"], F["best"]["GS"]
            md.append(f"| {basis} | token model · frame-set (canonical scores.json) | {F['unedited']['edit_index']:+.3f} | "
                      f"{fp['dims']}·pt{fp['point']}·α{fp['alpha']:g} · {fp['edit_index']:+.3f} / {fp['fidelity_ratio']:.2f} | "
                      f"{fg['dims']}·pt{fg['point']}·α{fg['alpha']:g} · {fg['edit_index']:+.3f} / {fg['fidelity_ratio']:.2f} |")
        if regS and basis in regS["bases"]:
            G = regS["bases"][basis]
            rp, rg = G["best"]["PI"], G["best"]["GS"]
            md.append(f"| {basis} | regression L-dw-8ray-20m · ray-zone (canonical) | {G['unedited']['edit_index']:+.3f} | "
                      f"{rp['dims']}·pt{rp['point']}·α{rp['alpha']:g} · {rp['edit_index']:+.3f} / {rp['fidelity_ratio']:.2f} | "
                      f"{rg['dims']}·pt{rg['point']}·α{rg['alpha']:g} · {rg['edit_index']:+.3f} / {rg['fidelity_ratio']:.2f} |")
    md += ["", f"UNK inputs seen by the adapter: {out['unk_inputs']} · {out['minutes']} min"]
    (EXP / "scores" / f"summary{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("done", path)


if __name__ == "__main__":
    main()
