"""Probe-capacity sweep: Probe Skill vs one-hidden-layer width, one residual point,
three sources (trained / random-init / observation), on a 5x probe corpus.

    python experiments/probe_capacity/scripts/probe_capacity.py discworld [--smoke]
    python experiments/probe_capacity/scripts/probe_capacity.py othello   [--smoke]

Question (Sevan, 2026-09-02): on discworld a random reservoir over the observation history
already reaches 0.96 of the trained model's 0.996 at MLP-128, on Othello 0.58 vs 0.98. Is
that a fixed fact about the two worlds, or does Othello's random floor also catch up once the
probe is wide enough — later than discworld's? At the canonical 20k/30k corpus a wide probe
memorises before it can show that, so the sweep runs on 5x the rows (discworld 250k
sequences, Othello 170k games ≈ 9.8M / 10M positions) and reports the in-sample gap.

⛔ Every fit is PERSISTED through ProbeCache before anything reads it; a rerun is cache hits.
⛔ Memory: the one-point residual stack is a memmap on the nvme (`.scratch/`), never RAM;
   fits stream from it (pim.probes.baselines.fit_probe_stream). Nothing here holds more than
   the observation tensor (~5 GB, on the GPU) at once.
After EVERY fit the scores JSON is rewritten atomically and outputs/probe_capacity.png is
re-rendered, so build_full_table's Fig 2 can be refreshed while this runs.
"""
import gc
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pim.environments.discworld import bench as dwb  # noqa: E402
from pim.figures.probe_capacity import capacity_figure  # noqa: E402
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.base import collect_residuals  # noqa: E402
from pim.probes.baselines import CausalHistory, MemmapRows, fit_probe_stream, random_init_model  # noqa: E402
from pim.probes.cache import ProbeCache  # noqa: E402

ENV = sys.argv[1]
SMOKE = "--smoke" in sys.argv
EXP = Path("experiments/probe_capacity")
OUT_JSON = EXP / "scores" / f"probe_capacity_{ENV}{'_smoke' if SMOKE else ''}.json"
PNG = EXP / "outputs" / f"probe_capacity{'_smoke' if SMOKE else ''}.png"
STORE = ProbeCache(EXP / "probes")
WIDTHS = [None, 16, 64, 128, 512, 1024, 2048]      # None = the linear probe
SOURCES = ("trained", "random_init", "observation")
EPOCHS, BATCH, SEED = (2 if SMOKE else 50), 4096, 0   # 50 epochs on 5x rows ≈ 2x the canonical step count
NT = "https://ntfy.sh/swirling-tornado-ai691k"
SCRATCH = Path(".scratch")
SCRATCH.mkdir(exist_ok=True)
DEV = dwb.DEV


def ping(title, body):
    try:
        req = urllib.request.Request(NT, data=body.encode(), headers={"Title": title})
        urllib.request.urlopen(req, timeout=20)
    except Exception:
        pass


def wlabel(w):
    return "LIN" if w is None else str(w)


def n_params(probe):
    return int(sum(p.numel() for p in probe.parameters()))


def write_scores(S):
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(S, indent=1, default=float))
    os.replace(tmp, OUT_JSON)                                   # atomic: readers never see a torn file
    fig = capacity_figure([EXP / "scores" / f"probe_capacity_{e}{'_smoke' if SMOKE else ''}.json"
                           for e in ("discworld", "othello")])
    fig.savefig(PNG, dpi=110, bbox_inches="tight")
    matplotlib.pyplot.close(fig)


# ── environment-specific loading ─────────────────────────────────────────────────────
if ENV == "discworld":
    RUN = Path("runs/initial_othello_comparison/L-dw-20m")
    root = Path("datasets/discworld/dw-pn04")
    corpus = root / "probe_250k" / "test.h5"
    N_SEQ = 400 if SMOKE else 250_000
    BASIS, TARGET, POINT = "frustum", "full", 3               # point 3 = canonical LIN/MLP argmax
    model, info = load_checkpoint(RUN / "best_model.pt", device=DEV)
    span = int(getattr(model, "state_span", 39))
    with h5py.File(corpus, "r") as f:
        obs = f["obs_intensity"][:N_SEQ].astype(np.float32)
        pos = f["positions"][:N_SEQ, :, :dwb.N_OBJ, :].astype(np.float32)
        vel = f["velocities"][:N_SEQ, :, :dwb.N_OBJ, :].astype(np.float32)
    sim = json.load(open(corpus.parent / "dataset.json"))["sim"]
    bp, bv = dwb._to_basis(pos, vel, sim, BASIS)
    y = np.concatenate([bp.reshape(N_SEQ, bp.shape[1], -1), bv.reshape(N_SEQ, bv.shape[1], -1)], -1)
    obs, y = obs[:, :span], y[:, :span]
    del pos, vel, bp, bv
    Y = torch.from_numpy(y).float().to(DEV)
    N_CLASSES, MASK = None, None
    T = obs.shape[1]
    canon = json.load(open(RUN / "scores.json"))["bases"][BASIS]
    base = json.load(open("runs/_baselines/dw-pn04/baselines.json"))["archs"][info.arch]["bases"][BASIS]
    refs = {"trained": {"LIN": canon["probe_skill_linear"][POINT], "128": canon["probe_skill_mlp"][POINT]},
            "random_init": {"LIN": base["random_init"]["linear"]["per_point"][POINT]["skill"],
                            "128": base["random_init"]["mlp"]["per_point"][POINT]["skill"]},
            "observation": {"LIN": base["observation"]["linear"]["skill"], "128": base["observation"]["mlp"]["skill"]}}
    refs_label, corpus_desc = "30k sequences", f"{corpus} n_seq={N_SEQ} span={span} basis={BASIS} target={TARGET}"

    def residual_source(m, tag):
        mm = SCRATCH / f"capacity_dw_{tag}.npy"
        R = collect_residuals(m, obs, batch=64, memmap=mm, points=[POINT])   # (1, N, T, d) on disk
        return MemmapRows(R[0], device=DEV), mm

    def observation_source():
        return CausalHistory(torch.from_numpy(obs).to(DEV))

else:
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.data import canonical_vocab, tokens_and_labels

    RUN = Path("runs/initial_othello_comparison/L-oth-20m")
    p = oc.build(only=("probe_large",))["probe_large"]
    tok, ln = oc.load(p)
    N_SEQ = 400 if SMOKE else len(tok)
    itos = {v: k for k, v in canonical_vocab().items()}
    data = tokens_and_labels([[itos[int(t)] for t in r[:L]] for r, L in zip(tok[:N_SEQ], ln[:N_SEQ])])
    model, info = load_checkpoint(RUN / "best_model.pt", device=DEV)
    S0 = json.load(open(RUN / "scores.json"))
    POINT = int(np.argmax(S0["probe_skill"]["mine|linear|sequence"]))   # canonical argmax
    BASIS, TARGET = "mine/theirs", "mine"
    Y = torch.from_numpy(data.mine.astype(np.int64)).to(DEV)
    N_CLASSES = 3
    MASK = torch.from_numpy(data.mask).to(DEV)
    T = data.tokens.shape[1]
    span = T
    base = json.load(open("runs/_baselines/oth-uniform/baselines.json"))["archs"][info.arch]["bases"][BASIS]
    st = {(x["family"]): x for x in S0["probe_stats"] if x["target"] == "mine" and x["split"] == "sequence" and x["point"] == POINT}

    def sk(x):
        return 1.0 - x["error_rate"] / x["majority_class_error_rate"]

    refs = {"trained": {"LIN": sk(st["linear"]), "128": sk(st["mlp"])},
            "random_init": {"LIN": base["random_init"]["linear"]["per_point"][POINT]["skill"],
                            "128": base["random_init"]["mlp"]["per_point"][POINT]["skill"]},
            "observation": {"LIN": base["observation"]["linear"]["skill"], "128": base["observation"]["mlp"]["skill"]}}
    refs_label, corpus_desc = "20k games", f"{p} n_games={N_SEQ} point={POINT}"
    VOCAB = len(canonical_vocab())

    def residual_source(m, tag):
        mm = SCRATCH / f"capacity_oth_{tag}.npy"
        out = None
        with torch.no_grad():
            for i in range(0, N_SEQ, 256):
                idx = torch.from_numpy(data.tokens[i:i + 256]).to(DEV)
                rs = m.residual_stack(idx)[POINT].float().cpu().numpy()          # (B, T, d)
                if out is None:
                    out = np.lib.format.open_memmap(str(mm), mode="w+", dtype=np.float32,
                                                    shape=(N_SEQ, rs.shape[1], rs.shape[2]))
                out[i:i + rs.shape[0]] = rs
        out.flush()
        return MemmapRows(out, device=DEV), mm

    def observation_source():
        return CausalHistory(torch.from_numpy(data.tokens).to(DEV), kind="one_hot", vocab=VOCAB)

perm = np.random.default_rng(SEED).permutation(N_SEQ)
tr, te = perm[: int(0.8 * N_SEQ)], perm[int(0.8 * N_SEQ):]
print(f"{ENV}: run {RUN.name}  point {POINT}  n_seq {N_SEQ}  T {T}  epochs {EPOCHS}  smoke={SMOKE}", flush=True)

S = {"env": ENV, "run": RUN.name, "arch": info.arch, "point": POINT, "basis": BASIS, "target": TARGET,
     "corpus": corpus_desc, "n_seq": N_SEQ, "split": "80/20 by sequence, seed 0", "epochs": EPOCHS,
     "batch": BATCH, "widths": [wlabel(w) for w in WIDTHS], "refs": refs, "refs_label": refs_label,
     "cells": {s: {} for s in SOURCES}, "started": time.strftime("%F %T")}
if OUT_JSON.exists():                                         # resume: keep finished cells
    prev = json.loads(OUT_JSON.read_text())
    S["cells"] = prev.get("cells", S["cells"])
write_scores(S)
ping(f"PIM capacity sweep: {ENV} started", f"{RUN.name} point {POINT}, n_seq {N_SEQ}, widths {S['widths']}")

for src in SOURCES:
    t_src = time.time()
    if src == "observation":
        hist, mm, key_model = observation_source(), None, None
        extra = {"span": span}
    else:
        m = model if src == "trained" else random_init_model(info.arch, info.model_config, seed=SEED, device=DEV)
        hist, mm = residual_source(m, src)
        key_model, extra = m, {}
        print(f"  [{src}] residual stack ready ({mm}, {mm.stat().st_size / 1e9:.1f} GB)", flush=True)
    for w in WIDTHS:
        lab = wlabel(w)
        if lab in S["cells"][src] and not SMOKE:
            print(f"  [{src}] width {lab}: already in scores — skip", flush=True)
            continue
        fname, prov = STORE.key(key_model, kind="capacity", env=ENV, run=RUN.name, source=src,
                                point=POINT, hidden=lab, n_seq=N_SEQ, epochs=EPOCHS, batch=BATCH,
                                seed=SEED, basis=BASIS, target=TARGET, corpus=corpus_desc, **extra)
        t0 = time.time()
        hit = STORE.load(fname, prov, device=DEV)
        if hit is not None:
            probe, st = hit
            print(f"  [{src}] width {lab}: cache HIT {fname}", flush=True)
        else:
            probe, st = fit_probe_stream(hist, Y, tr, te, hidden=w, n_classes=N_CLASSES,
                                         row_mask=MASK, seed=SEED, epochs=EPOCHS, batch=BATCH)
            STORE.store(fname, prov, (probe, st))                  # PERSISTED before use
            print(f"  [{src}] width {lab}: fit {time.time() - t0:.0f}s → WROTE {fname}", flush=True)
        skill = st["r2"] if "r2" in st else 1.0 - st["error_rate"] / st["majority_class_error_rate"]
        gap = (st["r2_insample"] - st["r2"] if "r2" in st
               else (st["error_rate"] - st["error_rate_insample"]) / st["majority_class_error_rate"])
        S["cells"][src][lab] = {"skill": float(skill), "insample_gap": float(gap), "params": n_params(probe),
                                "n_train_rows": st.get("n_train_rows"), "minutes": round((time.time() - t0) / 60, 2),
                                "cache": fname}
        write_scores(S)
        print(f"  [fit] {ENV} {src:<12} {lab:>5}: skill {skill:+.4f}  gap {gap:+.4f}  params {n_params(probe):,}", flush=True)
        del probe
        gc.collect()
        torch.cuda.empty_cache()
    row = "  ".join(f"{wlabel(w)}={S['cells'][src][wlabel(w)]['skill']:+.3f}" for w in WIDTHS if wlabel(w) in S["cells"][src])
    ping(f"PIM capacity {ENV}: {src} done ({(time.time() - t_src) / 60:.0f} min)", row)
    del hist
    if mm is not None and mm.exists():
        mm.unlink()                                                # the 20 GB stack: scratch, not a result
    gc.collect()
    torch.cuda.empty_cache()

S["finished"] = time.strftime("%F %T")
write_scores(S)
print(f"wrote {OUT_JSON} and {PNG}", flush=True)
