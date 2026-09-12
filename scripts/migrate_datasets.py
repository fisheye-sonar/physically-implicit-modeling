#!/usr/bin/env python
"""Migrate ``datasets/`` to layout v2 and re-key the probe caches — research/specs/DATASET_LAYOUT_SPEC.md.

    .pim/bin/python scripts/migrate_datasets.py --plan       # every move and cache rename; touches nothing
    .pim/bin/python scripts/migrate_datasets.py --snapshot   # inodes, bench hashes, cache provenance (before apply)
    .pim/bin/python scripts/migrate_datasets.py --apply      # renames (same filesystem), copies, markers, cache re-key
    .pim/bin/python scripts/migrate_datasets.py --verify     # CPU-only equivalence gate against the snapshot
    .pim/bin/python scripts/migrate_datasets.py --rollback   # reverse every rename and cache rename

Purely cosmetic by contract: every file keeps its inode; every cached probe keeps its bytes
and is re-keyed from its filesystem path onto the logical ``layout.probe_key``; nothing is
deleted (dead files go to ``<inst>/_unused/``, superseded duplicate cache blobs to
``<probes>/_superseded/``). Ledgers: ``datasets/MOVES.md``, ``runs/MOVES.md`` and the
machine-readable log in ``research/scratch/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pim.environments import layout  # noqa: E402

SCRATCH = _REPO / "research" / "scratch"
SNAPSHOT = SCRATCH / "2026-09-10-layout-snapshot.json"
LOG = SCRATCH / "2026-09-10-layout-migration-log.json"
DAY = "2026-09-10"
VENDORED_CASES = _REPO / "pim" / "environments" / "othello" / "vendor" / "intervention_benchmark.pkl"
OTH_N = {"train": 20_000_000, "test": 10_000, "probe": 20_000, "probe_large": 170_000}
SKIP_SNAPSHOT = ("_done_", "_shard_")          # transient generation markers under train/


def log(msg: str) -> None:
    print(msg, flush=True)


# ── discovery ────────────────────────────────────────────────────────────────


def instances(cls: str) -> list[str]:
    d = layout.DATASETS / cls
    return sorted(p.name for p in d.iterdir() if p.is_dir() and p.name != "archive") if d.exists() else []


def rel(p: Path) -> str:
    return str(p.relative_to(_REPO))


# ── the move tables (§2a / §2b of the spec) ──────────────────────────────────


def dw_plan(inst: str) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """(moves, copies) for a discworld instance, from the files that actually exist."""
    r = layout.instance_root("discworld", inst)
    moves, copies = [], []

    def mv(a, b):
        if (r / a).exists():
            moves.append((r / a, r / b))

    if (r / "eval" / "dataset.json").exists():       # one manifest described both surviving splits
        copies.append((r / "eval" / "dataset.json", r / "eval" / "test.json"))
    mv("eval/dataset.json", "edits/v1/edits.json")
    mv("eval/edits.h5", "edits/v1/edits.h5")
    mv("edits_selection.json", "edits/v1/selection.json")
    mv("eval/val.h5", "_unused/eval/val.h5")
    mv("eval/train.h5", "_unused/eval/train.h5")
    mv("probe/test.h5", "probe/probe_120k.h5")
    mv("probe/dataset.json", "probe/probe_120k.json")
    for s in ("train", "val", "edits"):
        mv(f"probe/{s}.h5", f"_unused/probe/{s}.h5")
    mv("probe_250k/test.h5", "probe/probe_250k.h5")
    mv("probe_250k/dataset.json", "probe/probe_250k.json")
    for s in ("train", "val", "edits"):
        mv(f"probe_250k/{s}.h5", f"_unused/probe_250k/{s}.h5")
    mv("tokens/probe.npy", "_unused/tokens/probe.npy")
    mv("tokens/val.npy", "_unused/tokens/val.npy")
    return moves, copies


def oth_plan(inst: str) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    r = layout.instance_root("othello", inst)
    moves, copies = [], []
    c = r / "corpus"
    if c.exists():
        for f in sorted(c.iterdir()):
            if not f.is_file():
                continue
            name = f.name
            dest = None
            for split, n in OTH_N.items():
                # the split itself and its FULL label cache; a partial cache (smoke) is dead
                if name in (f"{split}_{n}.npz", f"{split}_{n}_labels_{n}.npz"):
                    dest = r / layout.OTH_ROLE[split] / name
            if dest is None:                          # legacy rungs, smoke label caches, strays
                dest = r / "_unused" / "corpus" / name
            moves.append((f, dest))
    for ext in ("pkl", "json"):
        f = r / "edits" / f"cases_1001.{ext}"
        if f.exists():
            moves.append((f, r / "edits" / "v1" / f.name))
    if inst == "oth-uniform" and VENDORED_CASES.exists() and not (r / "edits" / "v1" / "cases_1001.pkl").exists():
        copies.append((VENDORED_CASES, r / "edits" / "v1" / "cases_1001.pkl"))
    return moves, copies


def full_plan() -> dict:
    plan = {}
    for inst in instances("discworld"):
        m, c = dw_plan(inst)
        plan[("discworld", inst)] = (m, c)
    for inst in instances("othello"):
        m, c = oth_plan(inst)
        plan[("othello", inst)] = (m, c)
    return plan


# ── probe caches ─────────────────────────────────────────────────────────────


def cache_dirs() -> list[Path]:
    out = set()
    for base in (_REPO / "runs", _REPO / "experiments"):
        for f in base.rglob("probes/probes_*.pt"):
            out.add(f.parent)
    return sorted(out)


def prov_fname(prov: dict) -> str:
    """The cache filename for a stored provenance dict — ``ProbeCache.key``'s hash, applied to
    the dict it stores (which IS the full key dict)."""
    h = hashlib.blake2b(repr(sorted(prov.items())).encode(), digest_size=8).hexdigest()
    return f"probes_{h}.pt"


def rekey(prov: dict) -> dict | None:
    """The re-keyed provenance, or None when the blob's corpus is not one that moves."""
    data = prov.get("data")
    if not isinstance(data, str):
        return None
    lk = layout.legacy_probe_key(data)
    if lk is None:
        return None
    new = dict(prov)
    new["data"], new["split"] = layout.probe_key(*lk)
    return new


def cache_plan() -> list[dict]:
    """[{dir, old, new, live}] — every blob that gets a new name. ``live`` marks the key the
    code currently produces (an ABSOLUTE path); a relative-path duplicate of the same fit
    (the pre-2026-09-01 double-fit) is superseded by the live one on collision."""
    import torch

    out = []
    for d in cache_dirs():
        for f in sorted(d.glob("probes_*.pt")):
            prov = torch.load(f, map_location="cpu", weights_only=False)["provenance"]
            if prov_fname(prov) != f.name:
                raise RuntimeError(f"{f}: stored provenance does not hash to its own filename — "
                                   f"the cache was tampered with; refusing to migrate")
            new = rekey(prov)
            if new is None:
                continue
            out.append({"dir": rel(d), "old": f.name, "new": prov_fname(new),
                        "live": os.path.isabs(prov["data"]), "old_data": prov["data"],
                        "new_data": new["data"], "new_split": new["split"]})
    return out


def apply_cache(entries: list[dict]) -> list[dict]:
    import torch

    from pim.probes.cache import ProbeCache

    done, by_dir = [], {}
    for e in entries:
        by_dir.setdefault(e["dir"], []).append(e)
    for d_rel, es in by_dir.items():
        d = _REPO / d_rel
        # collisions: several old blobs -> one new name; keep the live one, park the rest
        targets: dict[str, list[dict]] = {}
        for e in es:
            targets.setdefault(e["new"], []).append(e)
        for new, group in targets.items():
            group.sort(key=lambda e: (not e["live"], e["old"]))    # live first
            keep, park = group[0], group[1:]
            if (d / new).exists():
                raise RuntimeError(f"{d / new} already exists; refusing to overwrite")
            blob = torch.load(d / keep["old"], map_location="cpu", weights_only=False)
            blob["provenance"] = rekey(blob["provenance"])
            assert prov_fname(blob["provenance"]) == new
            tmp = (d / new).with_suffix(".pt.partial")
            torch.save(blob, tmp)
            tmp.replace(d / new)
            os.remove(d / keep["old"])
            done.append({**keep, "action": "rekeyed"})
            for p in park:
                sup = d / "_superseded"
                sup.mkdir(exist_ok=True)
                os.rename(d / p["old"], sup / p["old"])
                done.append({**p, "action": "superseded", "parked": rel(sup / p["old"])})
        ProbeCache(d).write_index()
    return done


# ── snapshot ─────────────────────────────────────────────────────────────────


def _hash_arrays(*arrs) -> str:
    import numpy as np

    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def bench_hash(inst: str, via: str) -> str:
    """A content hash of the canonical bench arrays, loaded the v2 way (``instance=``) or the
    legacy way (``data_dir=<inst>/eval``)."""
    from pim.environments.discworld import bench as dwb

    kw = {"instance": inst} if via == "instance" else {"data_dir": layout.instance_root("discworld", inst) / "eval"}
    a = dwb.bench_arrays(n=192, target="full", basis_name="frustum", **kw)
    z = a["zones"]
    return _hash_arrays(a["obs"], a["pos"], a["vel"], a["edit_object"], a["clean"], a["y"],
                        a["change_mask"], z.gt_edited, z.gt_unedited, z.differing, z.target,
                        z.ghost, z.collateral) + f"|n={a['n']}|sel={json.dumps(a['selection'], sort_keys=True)}"


def othello_bench_hash(inst: str) -> str:
    from pim.environments.othello.bench import load_benchmark

    b = load_benchmark(inst)
    return _hash_arrays(*b.tokens, *b.case_ids, b.pos_int, b.new_class, b.cur_lab, b.tgt_lab) + \
        "|" + hashlib.sha256(json.dumps([b.legal_pre, b.legal_post]).encode()).hexdigest()


def file_table(cls: str, inst: str) -> dict:
    r = layout.instance_root(cls, inst)
    out = {}
    for f in r.rglob("*"):
        if not f.is_file() or any(s in f.name for s in SKIP_SNAPSHOT):
            continue
        st = f.stat()
        out[str(f.relative_to(r))] = [st.st_ino, st.st_size]
    return out


def snapshot() -> dict:
    snap = {"taken": time.strftime("%Y-%m-%d %H:%M:%S"), "instances": {}, "caches": {},
            "vendored_cases_sha": hashlib.sha256(VENDORED_CASES.read_bytes()).hexdigest()}
    for cls in layout.CLASSES:
        for inst in instances(cls):
            rec = {"files": file_table(cls, inst), "migrated": layout.is_migrated(cls, inst)}
            if cls == "discworld" and layout.edits_file(cls, inst).exists():
                rec["bench_hash"] = bench_hash(inst, "instance")
                log(f"  bench {inst}: {rec['bench_hash'][:16]}…")
            if cls == "othello":
                if layout.othello_cases_file(inst).exists() or inst == "oth-uniform":
                    rec["bench_hash"] = othello_bench_hash(inst)
                    log(f"  bench {inst}: {rec['bench_hash'][:16]}…")
                rec["splits"] = {}
                for split, n in OTH_N.items():
                    p = layout.othello_split_file(inst, split, n)
                    if p.exists():
                        rec["splits"][split] = [p.stat().st_ino, p.stat().st_size]
            tok = layout.tokens_dir(inst) / "vocab.npz" if cls == "discworld" else None
            if tok is not None and tok.exists():
                rec["vocab_sha"] = hashlib.sha256(tok.read_bytes()).hexdigest()
            snap["instances"][f"{cls}/{inst}"] = rec
    import torch
    for d in cache_dirs():
        provs = {}
        for f in sorted(d.glob("probes_*.pt")):
            provs[f.name] = torch.load(f, map_location="cpu", weights_only=False)["provenance"]
        snap["caches"][rel(d)] = provs
    return snap


# ── preconditions ────────────────────────────────────────────────────────────


def preconditions(force: bool) -> None:
    problems = []
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        if gpu:
            problems.append(f"GPU compute processes present: {gpu}")
    except Exception as e:                                 # no nvidia-smi: not a blocker
        log(f"  (nvidia-smi unavailable: {e})")
    for unit in ("probe_targets_5", "probe_targets_4", "probe_targets_2", "probe_targets"):
        r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True)
        if r.stdout.strip() == "active":
            problems.append(f"systemd unit {unit} is active")
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    for needle in ("fit_probes.py", "nbconvert", "scripts/train.py", "master_eval"):
        if any(needle in line and "migrate_datasets" not in line for line in ps.splitlines()):
            problems.append(f"a process mentioning {needle} is running")
    if not SNAPSHOT.exists():
        problems.append(f"no snapshot at {rel(SNAPSHOT)} — run --snapshot first")
    if problems and not force:
        raise SystemExit("preconditions failed:\n  " + "\n  ".join(problems))
    for p in problems:
        log(f"  ⚠ forced past: {p}")


# ── apply ────────────────────────────────────────────────────────────────────


def apply(force: bool) -> None:
    preconditions(force)
    plan = full_plan()
    entries = cache_plan()
    logrec = {"applied": time.strftime("%Y-%m-%d %H:%M:%S"), "instances": {}, "caches": []}
    for (cls, inst), (moves, copies) in plan.items():
        r = layout.instance_root(cls, inst)
        for src, dst in moves:
            if dst.exists():
                raise SystemExit(f"refusing: destination exists {dst}")
        for src, dst in copies:
            if dst.exists() and not src.samefile(dst):
                raise SystemExit(f"refusing: copy destination exists {dst}")
    for (cls, inst), (moves, copies) in plan.items():
        r = layout.instance_root(cls, inst)
        for src, dst in copies:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            assert hashlib.sha256(src.read_bytes()).hexdigest() == hashlib.sha256(dst.read_bytes()).hexdigest()
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)                      # same filesystem: a metadata operation
        for empty in ("probe_250k", "corpus"):
            d = r / empty
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        rec_moves = [[str(s.relative_to(r)), str(d.relative_to(r))] for s, d in moves]
        rec_copies = [[rel(s) if not s.is_relative_to(r) else str(s.relative_to(r)), str(d.relative_to(r))]
                      for s, d in copies]
        layout.write_marker(cls, inst, moves=rec_moves, copies=rec_copies, spec="research/specs/DATASET_LAYOUT_SPEC.md")
        logrec["instances"][f"{cls}/{inst}"] = {"moves": rec_moves, "copies": rec_copies}
        log(f"  {cls}/{inst}: {len(moves)} moves, {len(copies)} copies, marker written")
    done = apply_cache(entries)
    logrec["caches"] = done
    LOG.write_text(json.dumps(logrec, indent=1))
    n_rk = sum(e["action"] == "rekeyed" for e in done)
    n_sp = sum(e["action"] == "superseded" for e in done)
    log(f"  probe caches: {n_rk} blobs re-keyed, {n_sp} duplicates parked under _superseded/")
    ledger(logrec)
    log(f"applied; log -> {rel(LOG)}")


def ledger(logrec: dict) -> None:
    lines = [f"\n## Layout v2 migration — {DAY} (`scripts/migrate_datasets.py --apply`; spec "
             f"`research/specs/DATASET_LAYOUT_SPEC.md`; log `{rel(LOG)}`)",
             "# Role-named splits: train/ probe/ eval/ edits/v1/ ; dead files -> _unused/ (kept). Per instance:"]
    for key, rec in logrec["instances"].items():
        lines.append(f"- `{key}`:")
        for s, d in rec["moves"]:
            lines.append(f"  - `{s}` → `{d}`")
        for s, d in rec["copies"]:
            lines.append(f"  - copy `{s}` → `{d}`")
    with open(layout.DATASETS / "MOVES.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    by_dir: dict[str, list] = {}
    for e in logrec["caches"]:
        by_dir.setdefault(e["dir"], []).append(e)
    lines = [f"\n## Probe-cache re-key — {DAY} (layout v2: cache keys name the corpus logically, "
             f"`data=discworld/<inst>`, `split=probe_<size>`, instead of a filesystem path; bytes unchanged; "
             f"full old→new list in `{rel(LOG)}`)"]
    for d, es in sorted(by_dir.items()):
        n_rk = sum(e["action"] == "rekeyed" for e in es)
        n_sp = sum(e["action"] == "superseded" for e in es)
        lines.append(f"- `{d}`: {n_rk} re-keyed" + (f", {n_sp} relative-path duplicates parked in `_superseded/`" if n_sp else ""))
    with open(_REPO / "runs" / "MOVES.md", "a") as f:
        f.write("\n".join(lines) + "\n")


# ── verify ───────────────────────────────────────────────────────────────────


def verify() -> None:
    import torch

    from pim.probes.cache import ProbeCache

    snap = json.loads(SNAPSHOT.read_text())
    logrec = json.loads(LOG.read_text())
    fails = []

    def check(ok: bool, msg: str) -> None:
        log(("  ✓ " if ok else "  ✗ ") + msg)
        if not ok:
            fails.append(msg)

    # 1. every snapshot file is present, same inode and size, at its (possibly new) path
    for key, rec in snap["instances"].items():
        cls, inst = key.split("/")
        r = layout.instance_root(cls, inst)
        marker = layout.read_layout(cls, inst) or {}
        mv = {s: d for s, d in marker.get("moves", [])}
        missing, changed = [], []
        for relp, (ino, size) in rec["files"].items():
            newp = r / mv.get(relp, relp)
            if not newp.exists():
                missing.append(relp)
            else:
                st = newp.stat()
                if st.st_ino != ino or st.st_size != size:
                    changed.append(relp)
        check(not missing and not changed, f"{key}: {len(rec['files'])} files present, same inodes and sizes"
              + (f" — MISSING {missing[:5]} CHANGED {changed[:5]}" if missing or changed else ""))
        check(layout.is_migrated(cls, inst), f"{key}: layout.json says v2")
        # nothing deleted: new files are only the marker and the copies
        now = file_table(cls, inst)
        extra = set(now) - {mv.get(k, k) for k in rec["files"]}
        allowed = {"layout.json"} | {d for _, d in marker.get("copies", [])}
        check(extra <= allowed, f"{key}: no unexpected files (extra: {sorted(extra - allowed)[:5]})")
        check(len(now) >= len(rec["files"]), f"{key}: file count {len(rec['files'])} → {len(now)} (nothing lost)")
        # 2. content hashes through the code, both the new and the legacy call form
        if "bench_hash" in rec:
            if cls == "discworld":
                h_new, h_old = bench_hash(inst, "instance"), bench_hash(inst, "data_dir")
                check(h_new == rec["bench_hash"], f"{key}: bench arrays identical via instance=")
                check(h_old == rec["bench_hash"], f"{key}: bench arrays identical via legacy data_dir=<inst>/eval")
            else:
                check(othello_bench_hash(inst) == rec["bench_hash"], f"{key}: Othello benchmark identical")
        if cls == "othello" and rec.get("splits"):
            from pim.environments.othello import corpus as oc
            paths = oc.build(only=tuple(rec["splits"]), instance=inst, log=lambda s: None)
            for split, (ino, size) in rec["splits"].items():
                st = paths[split].stat()
                check(st.st_ino == ino and st.st_size == size, f"{key}: corpus.build resolves {split} to the same file")
        if "vocab_sha" in rec:
            check(hashlib.sha256((layout.tokens_dir(inst) / "vocab.npz").read_bytes()).hexdigest() == rec["vocab_sha"],
                  f"{key}: tokens/vocab.npz unchanged")
            val = layout.unused_dir(cls, inst) / "eval" / "val.h5"
            if val.exists():
                import h5py
                from pim.environments.discworld.tokens import UNK, FrameVocab, encode
                voc = FrameVocab.load(layout.tokens_dir(inst) / "vocab.npz")
                with h5py.File(val) as h:
                    ids = encode(h["obs_intensity"][:], voc)
                check(bool((ids != UNK).all()), f"{key}: every frame of the retired eval/val.h5 is in the stored vocab (0 UNK)")
    # 3. caches: counts, self-consistent names, loadable through ProbeCache
    parked = {}
    for e in logrec["caches"]:
        if e["action"] == "superseded":
            parked[e["dir"]] = parked.get(e["dir"], 0) + 1
    for d_rel, provs in snap["caches"].items():
        d = _REPO / d_rel
        files = sorted(d.glob("probes_*.pt"))
        check(len(files) == len(provs) - parked.get(d_rel, 0),
              f"{d_rel}: {len(provs)} blobs → {len(files)} (+{parked.get(d_rel, 0)} parked)")
        store = ProbeCache(d)
        bad = []
        for f in files:
            prov = torch.load(f, map_location="cpu", weights_only=False)["provenance"]
            if prov_fname(prov) != f.name or store.load(f.name, prov) is None:
                bad.append(f.name)
            if isinstance(prov.get("data"), str) and layout.legacy_probe_key(prov["data"]) is not None \
                    and os.sep in prov["data"] and prov["data"].startswith(("/", "datasets")):
                bad.append(f"{f.name} still path-keyed: {prov['data']}")
        check(not bad, f"{d_rel}: every blob hashes to its name, loads, and is path-free ({bad[:3]})")
    # 4. the CODE's keys hit: every scored discworld run's canonical probes through probe_recipe
    hits, misses = 0, []
    from pim.environments.discworld import arms as dwa
    from pim.environments.discworld.grid_target import categorical_target
    from pim.models import load_checkpoint
    for cfg_path in sorted((_REPO / "runs").rglob("config.json")):
        run_dir = cfg_path.parent
        parts = run_dir.relative_to(_REPO / "runs").parts
        # the scorer's own exclusions (master_eval scan_runs): archive/ and "_" topics — the
        # smoke runs there were probed with smoke-sized recipes the canonical one cannot name
        if parts[0] == "archive" or parts[0].startswith("_") \
                or not (run_dir / "scores.json").exists() or not (run_dir / "best_model.pt").exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("data", {}).get("env") != "discworld":
            continue
        inst = cfg["data"]["instance"]
        scores = json.loads((run_dir / "scores.json").read_text())
        model, info = load_checkpoint(run_dir / "best_model.pt", device="cpu")
        enc = {}
        if info.arch.endswith("_tokens"):
            from pim.environments.discworld import token_bench as tkb
            from pim.environments.discworld.tokens import FrameVocab
            _e, _tag = tkb.token_encoder(FrameVocab.load(run_dir / "vocab.npz"))
            enc = {"encoder": _e, "encoder_tag": _tag}
        for key in scores.get("bases", {}):
            cat = categorical_target(key)
            target, basis = (key, "frustum") if cat else (scores.get("target", "full"), key)
            rec = dwa.probe_recipe(target, inst, n_seq=30_000)
            for fam in ("linear", "mlp"):
                try:
                    dwa.fit_probes(model, target=target, family=fam, basis_name=basis, cache_dir=run_dir / "probes",
                                   log=None, require_cached=True, **rec, **enc)
                    hits += 1
                except RuntimeError:
                    misses.append(f"{'/'.join(parts)}:{key}/{fam}")
        del model
    check(not misses, f"run probes: {hits} canonical (target, basis, family) keys HIT through probe_recipe; misses {misses[:6]}")
    # 5. the observation floors in runs/_baselines through observation_probes
    ohits, omiss = 0, []
    for inst in instances("discworld"):
        pdir = _REPO / "runs" / "_baselines" / inst / "probes"
        if not pdir.exists():
            continue
        for size, n_seq, epochs in (("120k", 30_000, None), ("250k", 250_000, 50)):
            if not layout.probe_file("discworld", inst, size).exists():
                continue
            for basis in ("cartesian", "frustum"):
                for fam in ("linear", "mlp"):
                    for align in ("left", "right"):
                        try:
                            dwa.observation_probes(target="full", n_seq=n_seq, family=fam, basis_name=basis, span=39,
                                                   cache_dir=pdir, log=None, epochs=epochs, align=align,
                                                   require_cached=True, probe={"instance": inst, "size": size})
                            ohits += 1
                        except RuntimeError:
                            omiss.append(f"{inst}/{size}/{basis}/{fam}/{align}")
    check(ohits > 0, f"observation floors: {ohits} keys HIT, {len(omiss)} absent (absent = never fitted, e.g. right-aligned on some instances): {omiss[:4]}")
    log("\nVERIFY " + ("PASSED" if not fails else f"FAILED ({len(fails)})"))
    if fails:
        raise SystemExit(1)


# ── rollback ─────────────────────────────────────────────────────────────────


def rollback() -> None:
    import torch

    from pim.probes.cache import ProbeCache

    logrec = json.loads(LOG.read_text())
    for key, rec in logrec["instances"].items():
        cls, inst = key.split("/")
        r = layout.instance_root(cls, inst)
        for s, d in reversed(rec["moves"]):
            (r / s).parent.mkdir(parents=True, exist_ok=True)
            os.rename(r / d, r / s)
        for s, d in rec["copies"]:
            if (r / d).exists():
                os.remove(r / d)
        for empty in ("_unused/probe", "_unused/probe_250k", "_unused/eval", "_unused/tokens", "_unused/corpus",
                      "_unused", "edits/v1", "probe", "eval", "train", "edits"):
            d = r / empty
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        lf = layout.layout_file(cls, inst)
        if lf.exists():
            os.remove(lf)
        log(f"  {key}: rolled back")
    for e in logrec["caches"]:
        d = _REPO / e["dir"]
        if e["action"] == "rekeyed":
            blob = torch.load(d / e["new"], map_location="cpu", weights_only=False)
            prov = dict(blob["provenance"])
            prov["data"], prov["split"] = e["old_data"], "test"
            assert prov_fname(prov) == e["old"], (e, prov_fname(prov))
            blob["provenance"] = prov
            tmp = (d / e["old"]).with_suffix(".pt.partial")
            torch.save(blob, tmp)
            tmp.replace(d / e["old"])
            os.remove(d / e["new"])
        else:
            os.rename(_REPO / e["parked"], d / e["old"])
    for d in {_REPO / e["dir"] for e in logrec["caches"]}:
        ProbeCache(d).write_index()
    log("rolled back (ledgers in MOVES.md are append-only and keep the record)")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("plan", "snapshot", "apply", "verify", "rollback"):
        g.add_argument(f"--{m}", action="store_true")
    ap.add_argument("--force", action="store_true", help="apply despite failed preconditions")
    a = ap.parse_args()
    if a.plan:
        for (cls, inst), (moves, copies) in full_plan().items():
            print(f"\n{cls}/{inst}  (migrated: {layout.is_migrated(cls, inst)})")
            for s, d in moves:
                print(f"  mv   {rel(s)}\n    -> {rel(d)}")
            for s, d in copies:
                print(f"  cp   {rel(s)}\n    -> {rel(d)}")
        entries = cache_plan()
        by = {}
        for e in entries:
            by.setdefault((e["dir"], e["live"]), 0)
            by[(e["dir"], e["live"])] += 1
        print(f"\nprobe caches: {len(entries)} blobs to re-key")
        for (d, live), n in sorted(by.items()):
            print(f"  {d}: {n} ({'absolute' if live else 'RELATIVE — will be superseded on collision'})")
        news = {}
        for e in entries:
            news.setdefault((e["dir"], e["new"]), []).append(e["old"])
        coll = {k: v for k, v in news.items() if len(v) > 1}
        print(f"  collisions (duplicate fits keyed once by a relative path): {len(coll)}")
    elif a.snapshot:
        SCRATCH.mkdir(exist_ok=True)
        snap = snapshot()
        SNAPSHOT.write_text(json.dumps(snap, indent=1))
        n_files = sum(len(r["files"]) for r in snap["instances"].values())
        n_blobs = sum(len(v) for v in snap["caches"].values())
        log(f"snapshot: {len(snap['instances'])} instances, {n_files} files, {n_blobs} cached probes -> {rel(SNAPSHOT)}")
    elif a.apply:
        apply(a.force)
    elif a.verify:
        verify()
    elif a.rollback:
        rollback()


if __name__ == "__main__":
    main()
