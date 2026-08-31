"""PI injection on discworld, in the SAME space Othello solves it in — plus the y-affine fix.

Scratch runner (2026-08-31), to be folded into `editability.py` once the housecleaning settles
where it belongs. Answers one question: how much of discworld's weak, destructive PI injection is
an artefact of *how the pseudoinverse was taken* rather than a fact about the world?

Three arms, so the two effects separate:

  A  legacy      raw-space pinv, y-affine DROPPED   — reproduces the published number
  B  raw+affine  raw-space pinv, y-affine included  — isolates the affine bug
  C  z+affine    z-space  pinv, y-affine included   — what `othello_transfer` does

The probe is `(A_z z + b_z) * y_std + y_mean` with `z = (h - x_mean) / x_std`.

* `editability._decompose` builds `W = A_z.T / x_std` and `bb = b_z - A_z (x_mean/x_std)`, i.e.
  `h @ W + bb == A_z z + b_z`. That is the **standardised-y** read-out, but `Bench.tgt` is in RAW
  sim units, so arm A solves for the wrong vector. On the cartesian probes the depth row has
  `y_mean 7.93, y_std 1.91`, a ~4.2 sigma offset — which is why arm A's best alpha pins at 175,
  the top of the sweep.
* Othello never hit this: its probes are classification (`n_classes=3`), where `fit_probe` sets
  `y_mean=0, y_std=1`, so the affine is the identity by construction.

`A_pinv` returns the minimum-norm write **in whatever coordinates it is taken**, so B and C differ
even though both are correct: B minimises |dh| in raw activation units (variance-blind), C
minimises |dh / x_std| — the write measured in activation standard deviations, which is the
on-manifold choice.

usage: pi_zspace.py [basis,csv] [n_probe] [probe_data_dir]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_HERE), str(_HERE.parent / "othello_gpt"), str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import editability as E  # noqa: E402
from model import build as build_arch  # noqa: E402
from pim.editors.probe_steering import inject_state  # noqa: E402

RUN = "runs/discworld_scale/BIG20M_discworld_L"
# One grid for all three arms so no arm is grid-limited: A needs the top (it pins at 175),
# B and C should land near 1 if the affine fix works.
ALPHAS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0,
          60.0, 100.0, 175.0)


def _maps(probe):
    """The three decompositions, as (A, A_pinv, b) triples ready for `inject_state`.

    Returns raw-space (D,H) operators for A/B and the z-space (D,H) operator for C, each with
    the pseudoinverse taken in its OWN space — that difference is the whole point of arm C.
    """
    lin = probe.net
    Az, bz = lin.weight.detach(), lin.bias.detach()          # (D,H), (D,)
    xs, xm = probe.x_std, probe.x_mean
    ys, ym = probe.y_std, probe.y_mean

    # A — legacy: standardised-y read-out, raw-space pinv (what `_decompose` returns)
    A_leg = Az / xs                                          # (D,H)
    b_leg = bz - (Az / xs) @ xm
    # B — same map, y-affine restored so the target is in the units `Bench.tgt` is in
    A_raw = ys[:, None] * A_leg
    b_raw = ym + ys * b_leg
    # C — z-space: solve against `net`'s own output, then map the write back through x_std
    return {
        "A legacy (raw pinv, no y-affine)": ("raw", A_leg, torch.linalg.pinv(A_leg), b_leg),
        "B raw pinv + y-affine": ("raw", A_raw, torch.linalg.pinv(A_raw), b_raw),
        "C z-space pinv + y-affine": ("z", Az, torch.linalg.pinv(Az), bz),
    }


@torch.no_grad()
def pi_arms(model, b: E.Bench, probes: dict, log=print) -> list[dict]:
    recs = []
    for ell, (probe, _) in probes.items():
        E.as_activations(model, ell)
        h0 = model.flat_state(b.state)
        z0 = (h0 - probe.x_mean) / probe.x_std
        for name, (space, A, Ap, bv) in _maps(probe).items():
            if space == "raw":
                h1 = inject_state(h0, b.tgt, A, Ap, bv)
                step = h1 - h0
                # read-out error is only comparable within an arm; quote it in the arm's units
                err0 = float(((h0 @ A.T + bv) - b.tgt).norm(dim=1).mean())
            else:
                tgt_net = (b.tgt - probe.y_mean) / probe.y_std
                z1 = inject_state(z0, tgt_net, A, Ap, bv)
                step = (z1 - z0) * probe.x_std
                err0 = float(((z0 @ A.T + bv) - tgt_net).norm(dim=1).mean())
            for a in ALPHAS:
                h = h0 + a * step
                roll = model.rollout_with_edit(b.state, ell, h, E.K_ROLL).cpu().numpy()
                recs.append({"editor": name, "point": ell, "alpha": a,
                             "write_ratio": float(
                                 ((a * step).norm(dim=1) / h0.norm(dim=1)).mean()),
                             "readout_err_before": err0,
                             **E.score(model, b, roll)})
    return recs


def run(basis_name: str, n_probe: int, probe_data_dir: str | None, log=print) -> dict:
    t0 = time.time()
    ck = torch.load(_REPO / RUN / "best_model.pt", map_location=E.DEV, weights_only=False)
    mc = ck["model_config"]
    m = build_arch(obs_res=mc["obs_res"], block_size=mc["block_size"],
                   n_layer=mc.get("n_layer", 8), n_head=mc.get("n_head", 8),
                   n_embd=mc.get("n_embd", 512), dropout=mc.get("dropout", 0.1)).to(E.DEV)
    m.load_state_dict(ck["model_state"])
    m.eval()
    log(f"{RUN}: {E.n_points(m)} residual points, val {ck['val_loss']:.5f}  basis={basis_name}")
    out = {"run": RUN, "basis": basis_name, "n_probe": n_probe, "targets": {}}

    for target in ("pos", "full"):
        b = E.load_bench(m, n=192, target=target, basis_name=basis_name)
        lin = E.fit_probes(m, target=target, n_seq=n_probe, hidden=None, log=None,
                           basis_name=basis_name, data_dir=probe_data_dir)
        p0 = lin[0][0]
        log(f"  target={target}  y_mean {np.round(p0.y_mean.cpu().numpy(), 3)}  "
            f"y_std {np.round(p0.y_std.cpu().numpy(), 3)}")
        u = E.unsteered(m, b)
        recs = pi_arms(m, b, lin, log=log)
        for r in recs:
            r["fidelity_ratio"] = E.fidelity_ratio(r, u)
        out["targets"][target] = {
            "unedited": {k: v for k, v in u.items() if np.isscalar(v)},
            "arms": [{k: v for k, v in r.items() if np.isscalar(v)} for r in recs],
        }
        log(f"  UNEDITED EI {u['edit_index']:+.4f}  target {u['target_rmse']:.4f}  "
            f"collat {u['collateral_rmse']:.4f}")
        log(f"  {'arm':<34}{'pt':>4}{'alpha':>8}{'EI':>9}{'fid':>7}"
            f"{'target':>9}{'collat':>9}{'|dh|/|h|':>10}")
        for ed in sorted({r["editor"] for r in recs}):
            sub = [r for r in recs if r["editor"] == ed]
            bst = max(sub, key=lambda r: r["edit_index"])
            log(f"  {ed:<34}{bst['point']:>4}{bst['alpha']:>8}{bst['edit_index']:>+9.4f}"
                f"{bst['fidelity_ratio']:>7.3f}{bst['target_rmse']:>9.4f}"
                f"{bst['collateral_rmse']:>9.4f}{bst['write_ratio']:>10.3f}")
    (_REPO / "runs" / "othello_arch"
     / f"BIG20M_discworld_L_pi_zspace_{basis_name}_p{n_probe}.json").write_text(
        json.dumps(out, indent=1, default=float))
    log(f"  [{(time.time() - t0) / 60:.1f} min]")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    bases = a[0].split(",") if a else ["cartesian", "inv_y"]
    n_probe = int(a[1]) if len(a) > 1 else 30000
    pdd = a[2] if len(a) > 2 and a[2] != "-" else "datasets/21_dwscale_probe"
    for bn in bases:
        print(f"\n{'=' * 96}\nBASIS: {bn}   PROBE SEQUENCES: {n_probe:,}\n{'=' * 96}", flush=True)
        run(bn, n_probe, pdd)
