#!/usr/bin/env python
"""Observation-space decodability floors for TOKENISED discworld (dw-8ray as tokens).

The canonical observation floor (`bench.observation_probes`) reads the causal FLOAT frame
history (39 x 8 = 312 features). For a frames-as-tokens model the input is a one-hot frame,
and its embedding is a linear map of that one-hot, so the fair LINEAR floor is a linear probe
on the one-hot history — any per-frame lookup is linear there. This fits, with the canonical
streamed baseline fit (`pim.probes.baselines.fit_probe_stream`, the same objective, split,
targets and stats as every Table 3 row):

  frame one-hot   CausalHistory(kind="one_hot", vocab=422): 39 x 422 = 16,458 features —
                  the model's literal input. LIN closed-form (streamed normal equations);
                  MLP-128 = 2.1M params, so it is fitted on the 30k corpus (0.6 rows/param,
                  expect memorisation) AND on the 250k corpus (4.6 rows/param, 50 epochs like
                  the canonical observation_large).
  ray one-hot     8 rays x 3 levels = 24 per frame, 936 features: lossless per ray, no
                  within-frame interaction. LIN + MLP on both corpora.

Every fitted probe is persisted under experiments/dw_tokens/obsfloor/probes/ (ProbeCache,
model=None, kind="observation_tokens"). References pulled beside them: the float floors and
the token architecture's random-init floor from runs/_baselines/dw-8ray/baselines.json, and
the trained token model's skills from its scores.json.

Output: scores/obs_floor_dw-8ray-tokens.json + scores/summary.md.
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

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from pim.environments.discworld.bench import N_OBJ, SEED, _to_basis  # noqa: E402
from pim.environments.discworld.tokens import FrameVocab, encode  # noqa: E402
from pim.probes import baselines as _bl  # noqa: E402
from pim.probes.baselines import CausalHistory, fit_probe_stream  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402
from pim.probes.mlp import CANONICAL_HIDDEN  # noqa: E402

EXP = REPO / "experiments" / "dw_tokens" / "obsfloor"


def _identity_moments(hist, s, f, chunk: int = 4096):
    """ONE-HOT features are not standardised (experiment-side override of the canonical
    streamed fit's input affine, 2026-09-06). The canonical `fit_probe_stream` divides every
    feature by its train std — right for residual activations, wrong for one-hots: a
    (position, frame) pair that is rare or absent in the train split has std ≈ 0, its
    standardised value on a held-out row is ~1e3–1e6, and both fits blow up (NaN linear
    solve, MLP skill −1e7 in the smoke). With the identity affine the linear closed form is
    plain least squares on the one-hot design (the intercept is in its span) and the MLP
    sees raw one-hots; everything else — split, targets, objective, stats — is canonical."""
    z = torch.zeros(hist.dim, device=hist.device)
    return z, torch.ones_like(z)


_bl._moments = _identity_moments


class RightAlignedHistory:
    """The causal history laid out RELATIVE TO THE PRESENT: block 0 = the current frame,
    block k = the frame k steps back, zero-padded beyond the start of the sequence.

    The canonical `CausalHistory` is LEFT-aligned (block j = frame j, zeroed after the
    present), so the current frame sits in a different block for every row and a LINEAR
    model cannot express even a current-frame lookup — measured 2026-09-06: the shared
    lookup alone reaches R² 0.968 (frustum), the left-aligned one-hot history linear fit
    0.726. Right alignment puts the lookup (and any fixed-lag read) inside the linear class.
    Same information, same rows, same split; only the layout differs. `kind` as in
    CausalHistory: "one_hot" (src = (N, T) ids, R = vocab) or "dense" (src = (N, T, R))."""

    def __init__(self, src: torch.Tensor, kind: str = "one_hot", vocab: int | None = None):
        self.src, self.kind, self.device = src, kind, src.device
        self.n, self.T = src.shape[0], src.shape[1]
        self.R = int(vocab) if kind == "one_hot" else int(src.shape[2])
        self.dim = self.T * self.R
        self._lag = torch.arange(self.T, device=src.device)

    def build(self, seq: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        idx = frame[:, None] - self._lag[None, :]                  # (B, T) source frame per block
        valid = idx >= 0
        g = self.src[seq[:, None].expand_as(idx), idx.clamp_min(0)]   # (B, T[, R])
        if self.kind == "one_hot":
            x = torch.zeros(len(seq), self.T, self.R, device=self.device)
            x.scatter_(2, g.unsqueeze(-1).long(), 1.0)
        else:
            x = g.float()
        x = x * valid.unsqueeze(-1).to(x.dtype)
        return x.reshape(len(seq), self.dim)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SPAN = 39


def load_split(dd: Path, n_seq: int, basis: str, vocab: FrameVocab):
    with h5py.File(dd / "test.h5", "r") as f:
        obs = f["obs_intensity"][:n_seq].astype(np.float32)
        pos = f["positions"][:n_seq, :, :N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:n_seq, :, :N_OBJ, :].astype(np.float32)
    sim = json.load(open(dd / "dataset.json"))["sim"]
    bp, bv = _to_basis(pos, vel, sim, basis)
    y = np.concatenate([bp.reshape(n_seq, bp.shape[1], -1), bv.reshape(n_seq, bv.shape[1], -1)], -1)
    tok = encode(obs, vocab)
    assert (tok > 0).all(), "a frame outside the vocabulary"
    return obs[:, :SPAN], tok[:, :SPAN], y[:, :SPAN]


def ray_onehot(obs: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """(N, T, R) frames -> (N, T, R*K) per-ray one-hot over the K levels."""
    idx = np.searchsorted(levels, obs)
    N, T, R = obs.shape
    out = np.zeros((N, T, R * len(levels)), np.float32)
    np.put_along_axis(out, (np.arange(R) * len(levels))[None, None, :] + idx, 1.0, axis=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", default="dw-8ray")
    ap.add_argument("--n-seq", type=int, default=30_000)
    ap.add_argument("--n-large", type=int, default=250_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--epochs-large", type=int, default=50)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_seq, a.n_large, a.epochs, a.epochs_large = 400, 800, 3, 2
    t0 = time.time()
    inst = REPO / "datasets" / "discworld" / a.instance
    vocab = FrameVocab.load(inst / "tokens" / "vocab.npz")
    V = vocab.size
    store = ProbeCache(EXP / "probes")
    res = {"instance": a.instance, "vocab_size": int(V), "span": SPAN, "rows": [], "smoke": a.smoke}

    corpora = [("30k", inst / "probe", a.n_seq, a.epochs), ("250k", inst / "probe_250k", a.n_large, a.epochs_large)]
    for basis in ("cartesian", "frustum"):
        for cname, dd, n_seq, epochs in corpora:
            obs, tok, y = load_split(dd, n_seq, basis, vocab)
            perm = np.random.default_rng(SEED).permutation(n_seq)
            tr, te = perm[: int(0.8 * n_seq)], perm[int(0.8 * n_seq):]
            yt = torch.from_numpy(y).float().to(DEV)
            tok_t = torch.from_numpy(tok).to(DEV)
            ray_t = torch.from_numpy(ray_onehot(obs, vocab.levels)).to(DEV)
            sources = {
                ("frame_onehot", "left"): CausalHistory(tok_t, kind="one_hot", vocab=V),
                ("ray_onehot", "left"): CausalHistory(ray_t),
                ("frame_onehot", "right"): RightAlignedHistory(tok_t, kind="one_hot", vocab=V),
                ("ray_onehot", "right"): RightAlignedHistory(ray_t, kind="dense"),
            }
            for (repr_name, align), hist in sources.items():
                for family in ("linear", "mlp"):
                    akey = {} if align == "left" else {"align": align}
                    hidden = None if family == "linear" else CANONICAL_HIDDEN
                    fname, prov = store.key(None, kind="observation_tokens", repr=repr_name, target="full",
                                            n_seq=int(n_seq), split="test", family=family, basis=basis,
                                            seed=SEED, span=SPAN, data=str(dd.resolve()), epochs=int(epochs),
                                            vocab=int(V), affine="identity", solver="pinv_hermitian", smoke=bool(a.smoke),
                                            **akey)
                    hit = store.load(fname, prov, device=DEV)
                    if hit is not None:
                        probe, st = hit
                        src = "cache"
                    else:
                        probe, st = fit_probe_stream(hist, yt, tr, te, hidden=hidden, seed=SEED,
                                                     epochs=epochs, log=None)
                        store.store(fname, prov, (probe, st))
                        src = "fit"
                    n_params = sum(p.numel() for p in probe.parameters())
                    row = {"basis": basis, "corpus": cname, "n_seq": int(n_seq), "repr": repr_name,
                           "align": align, "family": family, "epochs": int(epochs), "d_in": int(st["d_in"]),
                           "n_params": int(n_params), "n_train_rows": int(st["n_train_rows"]),
                           "rows_per_param": float(st["n_train_rows"] / n_params),
                           "skill": float(st["r2"]), "insample_gap": float(st["r2_insample"] - st["r2"]),
                           "per_dim_r2": st["per_dim_r2"], "source": src}
                    res["rows"].append(row)
                    print(f"  {basis:9s} {cname:4s} {repr_name:12s} {align:5s} {family:6s} d_in {row['d_in']:>6,} "
                          f"params {n_params:>9,} rows/param {row['rows_per_param']:8.1f}  "
                          f"skill {row['skill']:+.4f}  gap {row['insample_gap']:+.4f}  [{src}, {(time.time() - t0) / 60:.1f} min]",
                          flush=True)
            del sources, yt
            torch.cuda.empty_cache()

    # references: the float floors + random-init for the token arch, and the trained model
    refs = {}
    bp = REPO / "runs" / "_baselines" / a.instance / "baselines.json"
    sp = REPO / "runs" / "interface_ablation" / "L-dw-8ray-tok-20m" / "scores.json"
    if bp.exists():
        A = json.load(open(bp))["archs"].get("transformer_l_tokens", {}).get("bases", {})
        for basis, blk in A.items():
            refs[basis] = {"float_obs_30k": {f: blk["observation"][f]["skill"] for f in ("linear", "mlp")},
                           "float_obs_250k": {f: blk["observation_large"][f]["skill"] for f in ("linear", "mlp")},
                           "random_init_tokens": {f: blk["random_init"][f]["skill"] for f in ("linear", "mlp")}}
    if sp.exists():
        S = json.load(open(sp))
        for basis, T in S["bases"].items():
            refs.setdefault(basis, {})["trained_tokens"] = {"linear": max(T["probe_skill_linear"]),
                                                            "mlp": max(T["probe_skill_mlp"])}
    res["references"] = refs
    res["minutes"] = round((time.time() - t0) / 60, 1)
    tag = "_smoke" if a.smoke else ""
    (EXP / "scores").mkdir(exist_ok=True)
    out = EXP / "scores" / f"obs_floor_{a.instance}-tokens{tag}.json"
    out.write_text(json.dumps(res, indent=1))

    md = [f"# Observation floors for tokenised {a.instance} (span {SPAN}, full state, held out by sequence)", "",
          "| basis | features | corpus | LIN skill (gap) | MLP-128 skill (gap) | d_in | MLP params | rows/param (MLP) |",
          "|---|---|---|---|---|---|---|---|"]
    for basis in ("cartesian", "frustum"):
        for repr_name in ("frame_onehot", "ray_onehot"):
          for align in ("right", "left"):
            for cname in ("30k", "250k"):
                rr = {r["family"]: r for r in res["rows"] if r["basis"] == basis and r["repr"] == repr_name
                      and r["corpus"] == cname and r.get("align", "left") == align}
                if len(rr) < 2:
                    continue
                md.append(f"| {basis} | {repr_name} · {align}-aligned | {cname} | {rr['linear']['skill']:+.3f} ({rr['linear']['insample_gap']:+.3f}) | "
                          f"{rr['mlp']['skill']:+.3f} ({rr['mlp']['insample_gap']:+.3f}) | {rr['linear']['d_in']:,} | "
                          f"{rr['mlp']['n_params']:,} | {rr['mlp']['rows_per_param']:.1f} |")
        r = refs.get(basis, {})
        if r:
            md.append(f"| {basis} | float frames (canonical) | 30k / 250k | {r['float_obs_30k']['linear']:+.3f} / {r['float_obs_250k']['linear']:+.3f} | "
                      f"{r['float_obs_30k']['mlp']:+.3f} / {r['float_obs_250k']['mlp']:+.3f} | 312 | 41,096 | 28.5 / 237 |")
            md.append(f"| {basis} | random-init token model (best point) | 30k | {r['random_init_tokens']['linear']:+.3f} | {r['random_init_tokens']['mlp']:+.3f} | 512 | | |")
            if "trained_tokens" in r:
                md.append(f"| {basis} | **trained token model** (best point) | 30k | **{r['trained_tokens']['linear']:+.3f}** | **{r['trained_tokens']['mlp']:+.3f}** | 512 | | |")
    (EXP / "scores" / f"summary{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("done", out, f"[{res['minutes']} min]")


if __name__ == "__main__":
    main()
