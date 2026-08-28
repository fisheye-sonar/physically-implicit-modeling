"""Orchestration for `othello_transfer`: fit the probe grid, run the intervention arms.

The editing itself is **not** here. Every write is performed by
`../othello_gpt/othello_probe.py` — `build_edit_spec`, `make_intervention_hook`,
`_descend` — used unmodified, which is the point of the thread: if the edit fails on
this repo's own world model but works here, the failure is not in that code.

What this module owns is only the plumbing around it: which probes, which residual
points, which cases in which batch.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

THREAD = Path(__file__).resolve().parent
OTHELLO_ROOT = Path("/home/sevan/research/PIM/othello_world")
for _p in (str(THREAD), str(THREAD.parent / "othello_gpt"), str(OTHELLO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import othello_data as od  # noqa: E402
import othello_probe as op  # noqa: E402
from othello_shim import OthelloGPTShim  # noqa: E402

CKPT = THREAD.parents[3] / "runs" / "othello_transfer" / "gpt_synthetic.ckpt"
N_POINTS = 9  # residual points 0..8 for an 8-block transformer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Their unmodified `GPT`, behind our shim, in eval mode.

    `eval()` matters: minGPT carries dropout at p=0.1 and their own probe script never
    disables it (`train_probe_othello.py` has no `.eval()` call), so their harvest ran
    with dropout live. Ours does not.
    """
    from mingpt.model import GPT, GPTConfig

    gpt = GPT(GPTConfig(61, 59, n_layer=8, n_head=8, n_embd=512))
    gpt.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    return OthelloGPTShim(gpt.to(DEVICE).eval()).eval()


# ── probes ────────────────────────────────────────────────────────────────────


def _split(n_seq: int, seq_of_row: np.ndarray, how: str, holdout: float, seed: int):
    """Row indices for train/test.

    ``how="frame"`` reproduces Li's `random_split` over pooled (activation, board) rows —
    frames from one game land on both sides. ``how="sequence"`` holds out whole games,
    this repo's convention (`harness/ANALYSIS.md` §2). Numbers from the two are **not**
    comparable and are always reported side by side, labelled.
    """
    rng = np.random.default_rng(seed)
    n_rows = len(seq_of_row)
    if how == "frame":
        perm = rng.permutation(n_rows)
        cut = int(round((1 - holdout) * n_rows))
        return perm[:cut], perm[cut:]
    order = rng.permutation(n_seq)
    n_tr = int(round((1 - holdout) * n_seq))
    is_tr = np.zeros(n_seq, bool)
    is_tr[order[:n_tr]] = True
    tr_mask = is_tr[seq_of_row]
    return np.where(tr_mask)[0], np.where(~tr_mask)[0]


@dataclass
class ProbeGrid:
    probes: dict  # (target, family, split, point) -> WorldStateProbe
    stats: list  # one dict per fit


FAMILIES = {"MLP 512 hidden": 512, "MLP 128 hidden": 128, "linear": None}


CACHE_DIR = THREAD.parents[3] / "runs" / "othello_transfer" / "probe_cache"


def _cache_path(key: dict) -> Path:
    """One file per configuration. A single shared path let a 2-minute smoke run silently
    overwrite a 37-minute full-scale grid (2026-08-20); the key check caught it, but the
    work was still gone."""
    import hashlib

    h = hashlib.sha1(repr(sorted(key.items())).encode()).hexdigest()[:12]
    return CACHE_DIR / f"probe_grid_{h}.pt"


def fit_probe_grid(
    shim,
    data: od.ProbeData,
    *,
    targets=("state", "mine"),
    families=("MLP 512 hidden", "MLP 128 hidden", "linear"),
    splits=("frame", "sequence"),
    holdout: float = 0.2,
    epochs: int = 200,
    batch: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    log=print,
    cache: bool = True,
) -> ProbeGrid:
    """One probe per (target, family, split, residual point) — never shared across points.

    Harvests one residual point at a time (~2.4 GB at 20k games; all nine would be 22 GB).

    Fitting the full grid takes ~37 min, which would otherwise be paid again on every
    re-run of the downstream analysis. The grid is cached under a key covering every
    setting that changes a probe, so a changed setting refits rather than silently
    reusing the wrong probes.
    """
    key = {"targets": list(targets), "families": list(families), "splits": list(splits),
           "holdout": holdout, "epochs": epochs, "batch": batch, "lr": lr, "seed": seed,
           "n_seq": int(len(data.tokens)), "n_rows": int(data.mask.sum()),
           "n_points": N_POINTS}
    cache_path = _cache_path(key)
    if cache and cache_path.exists():
        blob = torch.load(cache_path, map_location=DEVICE, weights_only=False)
        if blob.get("key") == key:
            probes = {}
            for k, (d_in, d_out, hidden, sd) in blob["probes"].items():
                pr = op.WorldStateProbe(d_in, d_out, hidden, n_classes=od.N_CLASSES).to(DEVICE)
                pr.load_state_dict(sd)
                pr.eval()
                tgt, fam, split, point = k.split("|")
                probes[(tgt, fam, split, int(point))] = pr
            log(f"  loaded {len(probes)} probes from cache ({cache_path.name})")
            return ProbeGrid(probes, blob["stats"])
        log("  cached grid does not match this configuration — refitting")
    seq_of_row = np.repeat(np.arange(len(data.tokens))[:, None], data.tokens.shape[1], 1)[data.mask]
    ys = {
        "state": data.labels[data.mask].astype(np.int64),
        "mine": data.mine[data.mask].astype(np.int64),
    }
    idx_cache = {s: _split(len(data.tokens), seq_of_row, s, holdout, seed) for s in splits}

    probes, stats = {}, []
    for point in range(N_POINTS):
        acts = od.harvest_point(shim, data.tokens, point)
        x = acts[data.mask]
        del acts
        for target in targets:
            y = ys[target]
            for split in splits:
                tr, te = idx_cache[split]
                for fam in families:
                    probe, st = op.fit_probe(
                        x[tr], y[tr], x[te], y[te],
                        hidden=FAMILIES[fam], epochs=epochs, batch=batch, lr=lr,
                        device=DEVICE, seed=seed, n_classes=od.N_CLASSES,
                    )
                    st |= {"target": target, "family": fam, "split": split, "point": point}
                    probes[(target, fam, split, point)] = probe
                    stats.append(st)
                    log(f"  point {point}  {target:5s}  {split:8s}  {fam:14s}  "
                        f"error {st['error_rate']:6.2f}%  (in-sample {st['error_rate_insample']:5.2f}%)")
        del x
        torch.cuda.empty_cache()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"key": key, "stats": stats,
                    "probes": {"|".join(map(str, k)): (pr.d_in, pr.d_out, pr.hidden,
                                                       {n: t.cpu() for n, t in pr.state_dict().items()})
                               for k, pr in probes.items()}}, cache_path)
        log(f"  cached {len(probes)} probes to {cache_path.name}")
    return ProbeGrid(probes, stats)


# ── intervention ──────────────────────────────────────────────────────────────


def run_arm(
    shim,
    bench: od.Benchmark,
    probes: dict[int, op.WorldStateProbe],
    start_layer: int,
    *,
    alpha: float,
    n_steps: int,
    beta: float,
    optimizer: str = "adam",
) -> tuple[np.ndarray, dict]:
    """One intervention arm over all 1001 cases → (1001, 64) board probabilities.

    Bucket by bucket, because `make_intervention_hook` writes ``x[:, -1]``.
    Every write is theirs-via-ours: `build_edit_spec` + `make_intervention_hook`, untouched.
    """
    probs = np.zeros((bench.n_cases, od.N_TILES), np.float32)
    records: list[dict] = []
    for toks, ids in zip(bench.tokens, bench.case_ids):
        idx = torch.from_numpy(toks).to(DEVICE)
        b = len(ids)
        with torch.no_grad():
            rs = shim.residual_stack(idx)
        x0 = {ell: rs[ell][:, -1] for ell in range(N_POINTS)}
        cm = np.zeros((b, od.N_TILES), bool)
        cm[np.arange(b), bench.pos_int[ids]] = True
        tv = torch.zeros(b, od.N_TILES, dtype=torch.long, device=DEVICE)
        tv[torch.arange(b), torch.from_numpy(bench.pos_int[ids]).to(DEVICE)] = (
            torch.from_numpy(bench.new_class[ids]).to(DEVICE)
        )
        specs = {
            ell: op.build_edit_spec(probes[ell], x0[ell], cm, tv, beta=beta)
            for ell in range(N_POINTS)
        }
        rec: dict = {}
        hook = op.make_intervention_hook(
            probes, specs, start_layer,
            alpha=alpha, n_steps=n_steps, optimizer=optimizer, record=rec,
        )
        h, _ = shim._run(shim.embed(idx), None, edit=hook)
        with torch.no_grad():
            probs[ids] = od.board_probs(shim.decoder(shim.norm_out(h[:, -1])))
        records.append({"n": b, **{k: dict(v) for k, v in rec.items()}})
        del rs, x0, specs
    return probs, _merge_records(records)


def _merge_records(records: list[dict]) -> dict:
    """Case-weighted mean of the per-bucket read-out diagnostics."""
    total = sum(r["n"] for r in records)
    out: dict[int, dict] = {}
    for r in records:
        w = r["n"] / total
        for layer, d in r.items():
            if layer == "n":
                continue
            tgt = out.setdefault(layer, {})
            for k, v in d.items():
                tgt[k] = tgt.get(k, 0.0) + w * v
    return out


def unsteered(shim, bench: od.Benchmark) -> np.ndarray:
    """No intervention — the null arm, computed through the identical code path."""
    probs = np.zeros((bench.n_cases, od.N_TILES), np.float32)
    with torch.no_grad():
        for toks, ids in zip(bench.tokens, bench.case_ids):
            idx = torch.from_numpy(toks).to(DEVICE)
            probs[ids] = od.board_probs(shim.decode(idx))
    return probs
