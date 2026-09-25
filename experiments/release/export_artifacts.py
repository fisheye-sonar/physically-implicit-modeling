"""Export the public artifact bundle for the anonymous release.

Reads PRIVATE runs/ and datasets/ (read-only; never writes, moves or deletes there) and writes renamed,
scrubbed COPIES into STAGING, laid out as the release repo expects (runs/<env>/<variant>[__seedK]/,
runs/_baselines/<env>/<instance>/, datasets/<env>/<instance>/{eval,edits,probe,tokens}/).
Deterministic and re-runnable. Run from PRIVATE with PYTHONPATH=PRIVATE (probe pickles need its pim classes):

    cd PRIVATE && PYTHONPATH=$PWD .pim/bin/python experiments/release/export_artifacts.py all [--force]

Steps: export | manifest | verify | tables | lock   (all = every step, lock last; --force unlocks STAGING first).
Also shipped: experiments/release/generated/** (script outputs, same relative path) and each probes/INDEX.md,
rendered by the RELEASE ProbeCache.write_index in a RELEASE subprocess. Every Othello split (.npz) is the bytes the
RELEASE generator (pim.environments.othello.corpus.build) writes, rendered in a RELEASE subprocess with the source
games in place of generation, so a split the release scripts regenerate matches SHA256SUMS. Files a re-run no
longer ships are moved to REMOVED, never deleted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import pickletools
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

PRIVATE = Path("/home/sevan/research/PIM/physically-implicit-modeling")
STAGING = Path("/home/sevan/research/PIM/gms-release-artifacts")
WORK = PRIVATE / "experiments" / "release" / "work" / "export"
REPORTS = PRIVATE / "experiments" / "release" / "reports"
P_RUNS, P_DATA = PRIVATE / "runs", PRIVATE / "datasets"
PY = PRIVATE / ".pim" / "bin" / "python"
RELEASE = Path("/home/sevan/research/PIM/generative-models-as-simulators")
GENERATED = PRIVATE / "experiments" / "release" / "generated"      # script outputs shipped verbatim
REMOVED = PRIVATE / "experiments" / "release" / "work" / "fix-export" / "removed"
sys.path.insert(0, str(PRIVATE))

# ── names (SPEC "Names") ─────────────────────────────────────────────────────────────────────────────

MAIN_RUNS = [
    ("initial_othello_comparison/L-oth-20m", "othello/standard"),
    ("adjacent_flip_ablation/L-oth-adjacent-flip-20m", "othello/adjacent-flip"),
    ("adjacency_ablation/L-oth-adjacent-20m", "othello/adjacent-noflip"),
    ("flip_ablation/L-oth-noflip-20m", "othello/standard-noflip"),
    ("noise_ablation/L-dw-noiseless-20m", "rayworld/standard"),
    ("blink_ablation/L-dw-blink-20m", "rayworld/blink"),
    ("ray_ablation/L-dw-128ray-20m", "rayworld/128-ray"),
    ("ray_ablation/L-dw-16ray-20m", "rayworld/16-ray"),
    ("ray_ablation/L-dw-8ray-20m", "rayworld/8-ray"),
    ("ray_ablation/L-dw-5ray-20m", "rayworld/5-ray"),
]
EXTRA_RUNS = [
    ("smooth_ablation/L-dw-smooth-20m", "rayworld/smooth"),
    ("observer_ablation/L-dw-8ray-obs5-20m", "rayworld/obs5"),
    ("interface_ablation/L-dw-8ray-tok-20m", "rayworld/8-ray-tokens"),
]
REPLICATE_SUFFIX = [("__seed0_s512000", "__seed0"), ("__seed1", "__seed1"), ("__seed2", "__seed2")]

INSTANCES = {  # (old env, old instance) -> new instance
    ("othello", "oth-uniform"): "standard",
    ("othello", "oth-adjacent-flip"): "adjacent-flip",
    ("othello", "oth-adjacent"): "adjacent-noflip",
    ("othello", "oth-noflip"): "standard-noflip",
    ("discworld", "dw-noiseless"): "standard",
    ("discworld", "dw-blink"): "blink",
    ("discworld", "dw-128ray"): "128-ray",
    ("discworld", "dw-16ray"): "16-ray",
    ("discworld", "dw-8ray"): "8-ray",
    ("discworld", "dw-5ray"): "5-ray",
    ("discworld", "dw-smooth"): "smooth",
    ("discworld", "dw-8ray-obs5"): "obs5",
}
ENV_NEW = {"othello": "othello", "discworld": "rayworld"}
RAY_FAMILY = {"128-ray", "16-ray", "8-ray", "5-ray"}
EIGHT_RAY_MAIN_TARGETS = ("appearance-fac", "appearance", "grid-6x5", "grid-10x3", "grid-16x8", "pos@appearance")
# Blocks shipped only for the appendix qualitative grids (categorical PI / GS cells): main runs only, no IM arm,
# forward probes only; not a reported target, so baselines (floors) and replicates do not get them.
FIGURE_BLOCKS = {"rayworld/standard": ("appearance-fac",), "rayworld/blink": ("appearance-fac",)}

# Fallback for one in-scope PRIVATE file: runs/initial_othello_comparison/L-oth-20m/scores.json
# was deleted from the working tree at 22:14 by `git pull` (commit 2ebc669 untracked it while an older main
# still tracked it). The byte-identical copy the scoring host wrote (303,664 bytes, the size the artifact audit
# recorded at 21:36) was fetched read-only into WORK/recovered/; the PRIVATE file was restored (same sha256) at
# 22:42. The override is read only if the PRIVATE file is absent again.
SCORES_OVERRIDE = {
    "initial_othello_comparison/L-oth-20m": (
        WORK / "recovered" / "L-oth-20m.scores.json",
        "8e9fb261d5ba232e7e694e08bbac1b6d8e7cc4ac7a8eb798ef1208b3560e1bf6"),
}

# SPEC "Versions": every date-shaped version string -> "1.0"
VERSION_MAP = {v: "1.0" for v in ("2026-09-01.4", "2026-09-12.1", "2026-09-12.2", "2026-09-06.b4",
                                   "2026-09-15.1", "2026-09-19.1", "2026-09-23.1")}
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
VERSION_KEY = re.compile(r"(^|_)version$")
DROP_KEYS = {"created", "written", "generated_at", "commit_sha", "files_revised", "migrated"}
CATEGORICAL_STATE = "onehot-labels+cartesian-velocity"

# The TrainConfig fields every config.json train block holds (pim/training/train.py), in its order.
TRAIN_FIELDS = ("steps", "batch_size", "lr", "weight_decay", "grad_clip", "lr_schedule", "warmup_steps",
                "ckpt_base", "val_every", "seed")
# A legacy train block (othello/standard and its __seed0) records these too; no code reads them.
LEGACY_TRAIN_DROP = {"rung", "window", "arch", "epochs", "limit", "run_name", "val_fraction", "warmup_frac",
                     "d_model", "n_layers", "n_heads", "mlp_ratio"}
LEGACY_TRAIN = {"steps": 780000, "batch_size": 256, "lr": 0.001, "weight_decay": 0.0001, "grad_clip": 1.0,
                "lr_schedule": "constant", "warmup_steps": 2000, "ckpt_base": 1000, "val_every": 5000, "seed": 0}
LEGACY_TOP_DROP = ("rung", "w16_reference_steps")      # config.json top level
LEGACY_CKPT_DROP = ("rung",)                           # checkpoint dict top level
# Per-run files read by nothing (tables.probe_refit_spread reads variance.json of the other five runs).
UNREAD_RUN_FILES = {("othello/adjacent-flip", "variance.json"), ("rayworld/standard", "variance.json")}
OTHELLO_CASES_RECIPE = ("one occupied non-center token recolored; rejected if the legal set is unchanged or empty "
                        "(bench.synthesize_cases)")
OTHELLO_CASES_RECIPE_OLD = ("one occupied non-centre tile recoloured; rejected if the legal set is unchanged or "
                            "empty (bench.synthesise_cases)")
# replicate.note of every __seed0 member: the string RELEASE scripts/make_replicate_member.py writes.
REPLICATE_NOTE = "the main run's own checkpoint at the replicates' step budget"
REPLICATE_NOTE_OLD = "the canonical run's own checkpoint at the replicates' step budget"   # after strip_dates
# The shipped Othello splits: (corpus.build split name, path under datasets/othello/<instance>/).
OTHELLO_SPLITS = (("test", "eval/test_10000.npz"), ("probe", "probe/probe_20000.npz"),
                  ("probe_large", "probe/probe_large_170000.npz"))
# Blocks scored on the instance's selected cases but stored without the record: run -> (block, sibling block
# holding the record). Proof per block: the unedited card recomputed on selection.json's cases equals the
# block's, on the first n cases it does not (smooth: its selection is the first 1000 cases).
BENCH_SELECTION_FILL = {"rayworld/16-ray": ("frustum", "cartesian"), "rayworld/5-ray": ("frustum", "cartesian"),
                        "rayworld/8-ray": ("frustum", "cartesian"), "rayworld/blink": ("frustum", "cartesian"),
                        "rayworld/smooth": ("frustum", "cartesian"), "rayworld/standard": ("frustum", "cartesian")}
H5_DROP = "generated_at"                               # the one config_json key removed from every .h5

# Identity scan (verify step c): case-insensitive substrings; dates are checked separately.
IDENTITY = ["sevan", "brodjian", "caltech", "hobley", "perona", "/home/", "wsl", "lab box", "ntfy", "tailscale",
            "discworld", "dw-", "dw_", "l-dw", "l-oth", "oth-uniform", "oth-adjacent", "oth-noflip",
            "physically-implicit", "big20m", "w16_reference"]
# Raw-byte scan of every file (binary included): exact byte strings long enough not to occur by chance.
RAW_TOKENS = [b"sevan", b"Sevan", b"SEVAN", b"brodjian", b"Brodjian", b"caltech", b"Caltech", b"hobley", b"Hobley",
              b"perona", b"Perona", b"/home/", b"tailscale", b"discworld", b"Discworld", b"physically-implicit",
              b"oth-uniform", b"oth-adjacent", b"oth-noflip", b"BIG20M", b"generated_at", b"w16_reference"]
RAW_DATE = re.compile(rb"20\d\d-\d\d-\d\d")


def log(*a):
    print(*a, flush=True)


# ── string renames ───────────────────────────────────────────────────────────────────────────────────

def _inst_env_new(old_inst: str) -> tuple[str, str]:
    for (env, inst), new in INSTANCES.items():
        if inst == old_inst:
            return ENV_NEW[env], new
    raise KeyError(old_inst)


RUN_BARE = {old.split("/")[1]: new for old, new in MAIN_RUNS + EXTRA_RUNS}
_RUN_RE = re.compile(r"(?P<pre>runs/)?(?:(?P<topic>[a-z_]+)/)?(?P<run>L-(?:oth|dw)(?:-[A-Za-z0-9-]+?)?-20m)"
                     r"(?P<suf>__seed\d(?:_s\d+)?)?(?![\w-])")
_INST_RES = [(re.compile(p), n) for p, n in [
    (r"oth-adjacent-flip", "adjacent-flip"), (r"oth-adjacent(?![-\w])", "adjacent-noflip"),
    (r"oth-noflip", "standard-noflip"), (r"oth-uniform", "standard"), (r"dw-8ray-obs5", "obs5"),
    (r"dw-noiseless", "standard"), (r"dw-blink", "blink"), (r"dw-128ray", "128-ray"), (r"dw-16ray", "16-ray"),
    (r"dw-8ray(?![-\w])", "8-ray"), (r"dw-5ray", "5-ray"), (r"dw-smooth", "smooth")]]
_OLD_INST = r"(?:oth-adjacent-flip|oth-adjacent|oth-noflip|oth-uniform|dw-8ray-obs5|dw-8ray|dw-noiseless|dw-blink|dw-128ray|dw-16ray|dw-5ray|dw-smooth)"


def _run_sub(m: re.Match) -> str:
    suf = m.group("suf") or ""
    sufmap = dict(REPLICATE_SUFFIX)
    if suf and suf not in sufmap:
        raise ValueError(f"unmapped replicate suffix in {m.group(0)!r}")
    return (m.group("pre") or "") + RUN_BARE[m.group("run")] + (sufmap[suf] if suf else "")


def rename_str(s: str) -> str:
    """Old run / instance / environment names -> SPEC names; absolute prefixes -> repo-relative."""
    if "/datasets/" in s and s.startswith("/"):
        s = "datasets/" + s.split("/datasets/", 1)[1]
    if "/runs/" in s and s.startswith("/"):
        s = "runs/" + s.split("/runs/", 1)[1]
    s = re.sub(rf"datasets/(discworld|othello)/({_OLD_INST})/edits/v1(/|$)",
               lambda m: "datasets/{}/{}/edits{}".format(*_inst_env_new(m.group(2)), m.group(3)), s)
    s = re.sub(rf"datasets/(discworld|othello)/({_OLD_INST})(?![-\w])",
               lambda m: "datasets/{}/{}".format(*_inst_env_new(m.group(2))), s)
    s = re.sub(r"(datasets/(?:othello|rayworld)/[\w-]+)/corpus/", r"\1/train/", s)   # the release train/ split
    s = re.sub(rf"runs/_baselines/({_OLD_INST})(?![-\w])",
               lambda m: "runs/_baselines/{}/{}".format(*_inst_env_new(m.group(1))), s)
    s = _RUN_RE.sub(_run_sub, s)
    for rx, new in _INST_RES:
        s = rx.sub(new, s)
    s = s.replace("discworld", "rayworld").replace("Discworld", "Rayworld").replace("DISCWORLD", "RAYWORLD")
    return s


def strip_dates(s: str) -> str:
    s = re.sub(r",\s*20\d\d-\d\d-\d\d(?:[ T]\d\d:\d\d(?::\d\d)?)?", "", s)
    s = re.sub(r"\s*\(20\d\d-\d\d-\d\d(?:[ T]\d\d:\d\d(?::\d\d)?)?\)", "", s)
    return s


# ── JSON documents: structural edits (expected) then string edits (new) ──────────────────────────────

def is_nd(ed) -> bool:
    return isinstance(ed, str) and (ed == "ND" or ed.startswith("ND[") or ed.startswith("ND@"))


def strip_nd(o):
    """Remove every ND arm / ND entry anywhere below `o` (in place)."""
    if isinstance(o, dict):
        for k in [k for k in o if is_nd(k)]:
            del o[k]
        for v in o.values():
            strip_nd(v)
    elif isinstance(o, list):
        o[:] = [x for x in o if not (isinstance(x, dict) and is_nd(x.get("editor")))]
        for v in o:
            strip_nd(v)


def drop_keys(o, keys=DROP_KEYS):
    if isinstance(o, dict):
        for k in [k for k in o if k in keys]:
            del o[k]
        for v in o.values():
            drop_keys(v, keys)
    elif isinstance(o, list):
        for v in o:
            drop_keys(v, keys)


VERSIONS_APPLIED: Counter = Counter()


def string_pass(o, key=None):
    """Version map on version fields, then rename + date strip on every string leaf (returns new object)."""
    if isinstance(o, dict):
        return {k: string_pass(v, k) for k, v in o.items()}
    if isinstance(o, list):
        return [string_pass(v, key) for v in o]
    if isinstance(o, str):
        if key is not None and VERSION_KEY.search(str(key)) and DATE_RE.search(o):
            if o not in VERSION_MAP:
                raise ValueError(f"unmapped version string {o!r} under {key!r}")
            VERSIONS_APPLIED[o] += 1
            return VERSION_MAP[o]
        return strip_dates(rename_str(o))
    return o


def numeric_leaves(o, path="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            numeric_leaves(v, f"{path}/{k}", out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            numeric_leaves(v, f"{path}[{i}]", out)
    elif not isinstance(o, str):
        out[path] = o
    return out


def leaf_same(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return struct.pack("<d", a) == struct.pack("<d", b) or (math.isnan(a) and math.isnan(b))
    return a == b


def compare_numbers(expected, new) -> dict:
    """Every numeric leaf (int / float / bool / null) of `expected` is at the same path in `new`, bit-identical,
    and `new` has no numeric leaf `expected` lacks."""
    e, n = numeric_leaves(expected), numeric_leaves(new)
    missing = [p for p in e if p not in n]
    extra = [p for p in n if p not in e]
    diff = [p for p in e if p in n and not leaf_same(e[p], n[p])]
    return {"n": len(e), "missing": missing[:5], "extra": extra[:5], "diff": diff[:5],
            "ok": not (missing or extra or diff), "n_missing": len(missing), "n_extra": len(extra), "n_diff": len(diff)}


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def dump_json(obj) -> bytes:
    return (json.dumps(obj, indent=1) + "\n").encode()


# ── the run table ────────────────────────────────────────────────────────────────────────────────────

class Run:
    def __init__(self, old: str, new: str, kind: str, parent_new: str):
        self.old, self.new, self.kind, self.parent_new = old, new, kind, parent_new
        self.src = P_RUNS / old
        self.cfg = read_json(self.src / "config.json")
        self.env_old = self.cfg["data"]["env"]
        self.inst_old = self.cfg["data"]["instance"]
        self.env = ENV_NEW[self.env_old]
        self.inst = INSTANCES[(self.env_old, self.inst_old)]
        self.variant = parent_new.split("/")[1]
        self.arch = self.cfg["arch"]
        assert parent_new.split("/")[0] == self.env, (old, new)

    @property
    def replicate(self) -> bool:
        return self.kind == "replicate"

    @property
    def bases(self) -> tuple:
        if self.env != "rayworld":
            return ()
        return ("cartesian",) if self.inst == "obs5" else ("frustum", "cartesian")

    @property
    def extras(self) -> tuple:
        if self.env != "rayworld":
            return ()
        if self.new == "rayworld/8-ray":
            return EIGHT_RAY_MAIN_TARGETS
        if self.variant in RAY_FAMILY or self.variant == "8-ray-tokens":
            return ("appearance-fac",)
        return ()

    @property
    def run_extras(self) -> tuple:
        """`extras` plus FIGURE_BLOCKS (keyed by run id, so replicates get none); baseline scope uses `extras`."""
        return self.extras + tuple(t for t in FIGURE_BLOCKS.get(self.new, ()) if t not in self.extras)

    @property
    def keep_blocks(self) -> set:
        return set(self.bases) | set(self.run_extras)

    def scores_src(self) -> tuple[Path, str]:
        p = self.src / "scores.json"
        if p.exists():
            return p, "private"
        alt, sha = SCORES_OVERRIDE[self.old]
        got = sha256_file(alt)
        assert got == sha, f"recovered scores for {self.old} changed: {got}"
        return alt, "recovered"


def run_table() -> list[Run]:
    out = []
    for old, new in MAIN_RUNS:
        out.append(Run(old, new, "main", new))
    for old, new in EXTRA_RUNS:
        out.append(Run(old, new, "extra", new))
    for old, new in MAIN_RUNS:
        for s_old, s_new in REPLICATE_SUFFIX:
            out.append(Run(old + s_old, new + s_new, "replicate", new))
    return out


# ── hashing, copying ─────────────────────────────────────────────────────────────────────────────────

_SHA_CACHE_P = WORK / "sha_cache.json"
_SHA_CACHE: dict = {}


def sha256_file(p: Path) -> str:
    p = Path(p)
    st = p.stat()
    key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def load_sha_cache():
    global _SHA_CACHE
    if _SHA_CACHE_P.exists():
        _SHA_CACHE = read_json(_SHA_CACHE_P)


def save_sha_cache():
    _SHA_CACHE_P.write_text(json.dumps(_SHA_CACHE))


WRITTEN: dict[str, dict] = {}   # rel path -> {"src", "how"}


def _dest(rel: str) -> Path:
    d = STAGING / rel
    d.parent.mkdir(parents=True, exist_ok=True)
    return d


def put_bytes(rel: str, data: bytes, src: str, how: str):
    d = _dest(rel)
    if d.is_symlink():
        d.unlink()
    if not (d.exists() and d.stat().st_size == len(data) and d.read_bytes() == data):
        tmp = d.with_name(d.name + ".partial")
        tmp.write_bytes(data)
        os.replace(tmp, d)
    WRITTEN[rel] = {"src": src, "how": how}


def put_copy(rel: str, src: Path, how: str = "copy"):
    d = _dest(rel)
    src = Path(src)
    assert not src.is_symlink(), src
    if not (d.exists() and not d.is_symlink() and d.stat().st_size == src.stat().st_size
            and sha256_file(d) == sha256_file(src)):
        tmp = d.with_name(d.name + ".partial")
        shutil.copyfile(src, tmp)
        os.replace(tmp, d)
    WRITTEN[rel] = {"src": str(src.relative_to(PRIVATE)) if str(src).startswith(str(PRIVATE)) else str(src),
                    "how": how}


def private_rel(p: Path) -> str:
    return str(Path(p).relative_to(PRIVATE))


# ── fingerprints ─────────────────────────────────────────────────────────────────────────────────────

def state_fingerprint(state: dict) -> str:
    """pim.probes.cache.fingerprint, over a checkpoint's state dict (same keys and bytes as the loaded model)."""
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(state.items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def cache_fname(prov: dict) -> str:
    """pim.probes.cache.ProbeCache.key: the filename is a hash of the full provenance dict."""
    return f"probes_{hashlib.blake2b(repr(sorted(prov.items())).encode(), digest_size=8).hexdigest()}.pt"


def model_fingerprint(path: Path) -> str:
    from pim.models import load_checkpoint
    from pim.probes.cache import fingerprint
    m, _ = load_checkpoint(path, device="cpu")
    fp = fingerprint(m)
    del m
    return fp


_TOKTAG = {}


def tokens_tag(run: Run):
    if not run.arch.endswith("_tokens") or run.env != "rayworld":
        return None
    if run.old not in _TOKTAG:
        from pim.environments.discworld.token_bench import token_encoder
        from pim.environments.discworld.tokens import FrameVocab
        _TOKTAG[run.old] = token_encoder(FrameVocab.load(run.src / "vocab.npz"))[1]
    return _TOKTAG[run.old]


# ── probe scope (decided from each file's stored provenance) ─────────────────────────────────────────

def classify_run_probe(prov: dict, run: Run, fp: str, im_targets: set) -> tuple[bool, str]:
    if prov.get("model") != fp:
        return False, "fingerprint mismatch (stale fit of another checkpoint)"
    if prov.get("v") != 2:
        return False, f"cache version {prov.get('v')}"
    if int(prov.get("seed", 0)) != 0:
        return False, "probe-seed replicate (seed != 0)"
    kind, tgt = prov.get("kind"), prov.get("target")
    if run.env == "othello":
        if kind == "othello_grid":
            if (prov.get("targets") == ["mine"] and prov.get("families") == ["linear", "mlp"]
                    and prov.get("splits") == ["sequence"] and prov.get("n_seq") == 20000):
                return True, "grid mine (LIN+MLP)"
            return False, f"grid {prov.get('targets')} {prov.get('families')} {prov.get('splits')} n={prov.get('n_seq')}"
        if kind == "inverse_map" and tgt == "mine-onehot" and prov.get("n_games") == 20000:
            return True, "IM mine-onehot"
        return False, f"othello {kind} {tgt}"
    # rayworld
    if prov.get("data") != f"discworld/{run.inst_old}":
        return False, f"data {prov.get('data')!r}"
    if prov.get("encoder") != tokens_tag(run):
        return False, f"encoder {prov.get('encoder')!r}"
    bases, extras = run.bases, run.run_extras
    if kind is None:
        if tgt == "full":
            if (prov.get("basis") in bases and prov.get("n_seq") == 30000 and prov.get("split") == "probe_120k"
                    and "epochs" not in prov):
                return True, f"forward full/{prov['basis']}"
            return False, f"forward full/{prov.get('basis')} (basis or recipe not in scope)"
        if tgt == "pos@appearance":
            if (tgt in extras and prov.get("basis") == bases[0] and prov.get("n_seq") == 30000
                    and prov.get("split") == "probe_120k"):
                return True, "forward pos@appearance (snapped)"
            return False, "pos@appearance not in scope"
        if tgt in extras:
            if (prov.get("basis") == bases[0] and prov.get("n_seq") == 200000 and prov.get("split") == "probe_250k"
                    and prov.get("epochs") == 50):
                return True, f"forward categorical {tgt}"
            return False, f"categorical {tgt} other recipe"
        return False, f"target {tgt} not in scope"
    if kind == "inverse_map":
        if prov.get("state") == CATEGORICAL_STATE:
            if tgt in im_targets and prov.get("basis") == bases[0] and prov.get("n_seq") == 200000:
                return True, f"categorical IM {tgt}"
            return False, f"categorical IM {tgt} not in scope"
        if (tgt == "full" and "state" not in prov and prov.get("basis") in bases and prov.get("n_seq") == 30000
                and prov.get("hidden") == 128 and prov.get("epochs") == 200):
            return True, f"IM full/{prov['basis']}"
        return False, f"IM {tgt}/{prov.get('basis')} not in scope"
    return False, f"kind {kind}"


def probe_slot(prov: dict) -> tuple:
    return (prov.get("kind"), prov.get("target"), str(prov.get("targets")), prov.get("family"),
            str(prov.get("families")), prov.get("basis"), prov.get("point"), prov.get("align"),
            prov.get("epochs"), prov.get("n_seq"), prov.get("model"))


def rekey_prov(prov: dict) -> tuple[dict, list]:
    """SPEC: only the Rayworld `data` field changes (plus version strings, none are stored in probe keys)."""
    new, changed = dict(prov), []
    for k, v in prov.items():
        if k == "data" and isinstance(v, str) and v.startswith("discworld/"):
            env, inst = _inst_env_new(v.split("/", 1)[1])
            new[k] = f"{env}/{inst}"
            changed.append(k)
        elif isinstance(v, str) and VERSION_KEY.search(k) and DATE_RE.search(v):
            new[k] = VERSION_MAP[v]
            changed.append(k)
    for k, v in new.items():
        if isinstance(v, str):
            low = v.lower()
            assert not any(t in low for t in IDENTITY), (k, v)
    return new, changed


def _torch_load(p):
    import torch
    return torch.load(p, map_location="cpu", weights_only=False)


def deep_same(a, b) -> bool:
    import numpy as np
    import torch
    if type(a) is not type(b):
        return False
    if torch.is_tensor(a):
        return (a.dtype == b.dtype and a.shape == b.shape
                and a.detach().cpu().contiguous().numpy().tobytes() == b.detach().cpu().contiguous().numpy().tobytes())
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return list(a.keys()) == list(b.keys()) and all(deep_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(deep_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return leaf_same(a, b)
    if isinstance(a, (int, str, bool, type(None), bytes, np.generic)):
        return a == b or (isinstance(a, np.floating) and np.isnan(a) and np.isnan(b))
    if hasattr(a, "__dict__"):
        return deep_same(vars(a), vars(b))
    return a == b


def ship_probe(src: Path, rel_dir: str, rekey: bool) -> dict:
    """Copy one cache file (Othello: verbatim) or re-key it (Rayworld: new `data`, new filename)."""
    import torch
    if not rekey:
        prov = _torch_load_prov(src)
        assert cache_fname(prov) == src.name, f"hash does not reproduce {src}"
        new_prov, changed = rekey_prov(prov)
        assert not changed, (src, changed)
        rel = f"{rel_dir}/{src.name}"
        put_copy(rel, src, "copy (provenance unchanged)")
        return {"src": src.name, "dst": src.name, "changed": [], "bytes": src.stat().st_size}
    blob = _torch_load(src)
    prov = blob["provenance"]
    assert cache_fname(prov) == src.name, f"hash does not reproduce {src}"
    new_prov, changed = rekey_prov(prov)
    name = cache_fname(new_prov)
    rel = f"{rel_dir}/{name}"
    d = _dest(rel)
    blob2 = dict(blob)
    blob2["provenance"] = new_prov
    if not (d.exists() and _probe_matches(d, new_prov, blob)):
        tmp = d.with_suffix(".pt.partial")        # the same tmp -> replace path ProbeCache.store takes
        torch.save(blob2, tmp)
        tmp.replace(d)
        assert _probe_matches(d, new_prov, blob), f"re-saved payload differs: {d}"
    WRITTEN[rel] = {"src": private_rel(src), "how": f"re-keyed ({','.join(changed)})"}
    return {"src": src.name, "dst": name, "changed": changed, "bytes": d.stat().st_size}


def _probe_matches(d: Path, new_prov: dict, blob_old: dict) -> bool:
    b = _torch_load(d)
    if b.get("provenance") != new_prov or list(b.keys()) != list(blob_old.keys()):
        return False
    return all(deep_same(b[k], blob_old[k]) for k in blob_old if k != "provenance")


# ── step: runs ───────────────────────────────────────────────────────────────────────────────────────

NUMBER_CHECKS: list[dict] = []
PROBE_ACCOUNT: dict = {}
RUN_FILES_ACCOUNT: dict = {}
BLOCKS_SHIPPED: dict = {}


def transform_scores(old: dict, run: Run) -> tuple[dict, dict]:
    s = copy.deepcopy(old)
    for k in ("commit_sha", "settings", "blocks_added", "inverse_added"):
        s.pop(k, None)
    s["bases"] = {k: v for k, v in s.get("bases", {}).items() if k in run.keep_blocks}
    for b in s["bases"].values():
        b.pop("inverse_cleared", None)
    if run.new in BENCH_SELECTION_FILL:
        blk, sib = BENCH_SELECTION_FILL[run.new]
        assert s["bases"][blk]["bench_selection"] is None and s["bases"][sib]["bench_selection"], (run.new, blk)
        s["bases"][blk]["bench_selection"] = copy.deepcopy(s["bases"][sib]["bench_selection"])
    strip_nd(s)
    drop_keys(s)
    expected = copy.deepcopy(s)
    new = string_pass(s)
    new["run"] = run.new
    new["env"] = run.env
    new["instance"] = run.inst
    new["probe_dir"] = f"runs/{run.new}/probes"
    return new, expected


def is_legacy_train(tc: dict) -> bool:
    return "run_name" in tc


def train_fields(tc: dict) -> dict:
    """A train block as TrainConfig fields: run_dir dropped; a legacy block keeps only the fields it shares."""
    tc = {k: v for k, v in tc.items() if k != "run_dir"}
    if not is_legacy_train(tc):
        assert set(tc) <= set(TRAIN_FIELDS), sorted(set(tc) - set(TRAIN_FIELDS))
        return tc
    assert set(tc) == set(TRAIN_FIELDS) | LEGACY_TRAIN_DROP, sorted(set(tc) ^ (set(TRAIN_FIELDS) | LEGACY_TRAIN_DROP))
    new = {k: tc[k] for k in TRAIN_FIELDS}
    assert new == LEGACY_TRAIN and all(type(new[k]) is type(v) for k, v in LEGACY_TRAIN.items()), new
    return new


def transform_config(old: dict, run: Run) -> tuple[dict, dict]:
    c = copy.deepcopy(old)
    c.pop("commit_sha", None)
    if isinstance(c.get("train"), dict):
        if is_legacy_train(c["train"]):
            for k in LEGACY_TOP_DROP:
                c.pop(k)
        c["train"] = train_fields(c["train"])
    for rec in c.get("resumed", []):          # the training loop's resume record: keep steps, drop the timestamp
        rec.pop("at", None)
    drop_keys(c)
    expected = copy.deepcopy(c)
    new = string_pass(c)
    new["data"]["env"] = run.env
    new["data"]["instance"] = run.inst
    if "replicate" in new:
        assert new["replicate"]["of"] == run.parent_new, new["replicate"]
        if "note" in new["replicate"]:
            assert new["replicate"]["note"] in (REPLICATE_NOTE_OLD, REPLICATE_NOTE), new["replicate"]["note"]
            new["replicate"]["note"] = REPLICATE_NOTE
    return new, expected


def transform_generic(old, run_or_none=None) -> tuple:
    d = copy.deepcopy(old)
    strip_nd(d)
    drop_keys(d)
    expected = copy.deepcopy(d)
    return string_pass(d), expected


def record_number_check(rel: str, expected, new, source: str):
    r = compare_numbers(expected, new)
    r.update({"file": rel, "source": source})
    NUMBER_CHECKS.append(r)
    if not r["ok"]:
        raise AssertionError(f"number check failed for {rel}: {r}")


def export_runs(runs: list[Run]):
    import torch
    fps = {}
    for run in runs:
        t0 = time.time()
        rd = f"runs/{run.new}"
        acc = {"kept": [], "dropped": []}
        # checkpoint: re-save only where the train_config changes (run_dir; a legacy block -> TrainConfig fields)
        ck_src = run.src / "best_model.pt"
        ck = torch.load(ck_src, map_location="cpu", weights_only=False)
        fp_old = state_fingerprint(ck["model_state"])
        fps[run.new] = fp_old
        tc = ck.get("train_config") or {}
        tc_new = train_fields(tc) if tc else tc
        ck_drop = [k for k in LEGACY_CKPT_DROP if k in ck] if is_legacy_train(tc) else []
        if tc_new != tc or ck_drop:
            ck2 = {k: v for k, v in ck.items() if k not in ck_drop}
            ck2["train_config"] = tc_new
            if is_legacy_train(tc):
                assert tc_new == train_fields(run.cfg["train"]), run.new      # checkpoint and config.json agree
            d = _dest(f"{rd}/best_model.pt")
            tmpd = d.parent / ".partial"            # keeps the archive's inner name "best_model", as saved by training
            tmpd.mkdir(exist_ok=True)
            torch.save(ck2, tmpd / "best_model.pt")
            os.replace(tmpd / "best_model.pt", d)
            tmpd.rmdir()
            back = torch.load(d, map_location="cpu", weights_only=False)
            assert list(back["model_state"].keys()) == list(ck["model_state"].keys())
            assert all(deep_same(back["model_state"][k], ck["model_state"][k]) for k in ck["model_state"])
            assert state_fingerprint(back["model_state"]) == fp_old
            assert back["train_config"] == tc_new and list(back) == list(ck2)
            assert {k: v for k, v in back.items() if k not in ("model_state", "train_config")} == \
                   {k: v for k, v in ck.items() if k not in ("model_state", "train_config", *ck_drop)}
            WRITTEN[f"{rd}/best_model.pt"] = {
                "src": private_rel(ck_src),
                "how": "re-saved (train_config as TrainConfig fields" + (f"; dropped {ck_drop})" if ck_drop else ")")}
        else:
            put_copy(f"{rd}/best_model.pt", ck_src)
        del ck
        acc["kept"].append(("best_model.pt", ck_src.stat().st_size))
        # config.json
        cfg_new, cfg_exp = transform_config(run.cfg, run)
        put_bytes(f"{rd}/config.json", dump_json(cfg_new), private_rel(run.src / "config.json"), "transformed")
        record_number_check(f"{rd}/config.json", cfg_exp, cfg_new, private_rel(run.src / "config.json"))
        # scores.json
        sp, how = run.scores_src()
        old = read_json(sp)
        new, exp = transform_scores(old, run)
        put_bytes(f"{rd}/scores.json", dump_json(new), private_rel(sp), f"transformed ({how})")
        record_number_check(f"{rd}/scores.json", exp, new, private_rel(sp))
        BLOCKS_SHIPPED[run.new] = {"old_run": run.old, "blocks": list(new["bases"]),
                                   "dropped_blocks": [k for k in old.get("bases", {}) if k not in new["bases"]]}
        missing_blocks = sorted(run.keep_blocks - set(new["bases"]))
        if missing_blocks:
            BLOCKS_SHIPPED[run.new]["in_scope_but_absent"] = missing_blocks
        # other per-run files
        for name in ("metrics.jsonl", "vocab.npz"):
            if (run.src / name).exists():
                put_copy(f"{rd}/{name}", run.src / name)
        for name in ("variance.json", "editability_by_reachability.json", "two_flip_editability.json"):
            if (run.src / name).exists() and (run.new, name) in UNREAD_RUN_FILES:
                acc["dropped"].append((name + " (read by nothing)", du(run.src / name)))
            elif (run.src / name).exists():
                n2, e2 = transform_generic(read_json(run.src / name))
                put_bytes(f"{rd}/{name}", dump_json(n2), private_rel(run.src / name), "transformed")
                record_number_check(f"{rd}/{name}", e2, n2, private_rel(run.src / name))
        for p in sorted(run.src.iterdir()):
            if p.name not in ("best_model.pt", "config.json", "scores.json", "metrics.jsonl", "vocab.npz",
                              "variance.json", "editability_by_reachability.json", "two_flip_editability.json",
                              "probes"):
                acc["dropped"].append((p.name + ("/" if p.is_dir() else ""), du(p)))
        # probes
        im_targets = {k for k, b in new["bases"].items()
                      if b.get("kind") == "classification" and any(a.get("editor") == "IM" for a in b.get("arms", []))}
        pk, pd_ = [], []
        slots = {}
        pdir = run.src / "probes"
        for p in sorted(pdir.iterdir()):
            if p.is_dir() or not p.name.startswith("probes_") or p.suffix != ".pt":
                pd_.append({"file": p.name + ("/" if p.is_dir() else ""), "bytes": du(p),
                            "why": "INDEX.md (derived; dropped)" if p.name == "INDEX.md" else "not a cache file"})
                continue
            prov = _torch_load_prov(p)
            keep, why = classify_run_probe(prov, run, fp_old, im_targets)
            if not keep:
                pd_.append({"file": p.name, "bytes": p.stat().st_size, "why": why})
                continue
            sl = probe_slot(prov)
            assert sl not in slots, f"two cache files for one slot in {run.old}: {slots[sl]} {p.name}"
            slots[sl] = p.name
            info = ship_probe(p, f"{rd}/probes", rekey=(run.env == "rayworld"))
            info["why"] = why
            pk.append(info)
        cover = probe_coverage(run, new, [s for s in slots])
        PROBE_ACCOUNT[run.new] = {"old_run": run.old, "fingerprint": fp_old,
                                  "kept": len(pk), "kept_bytes": sum(x["bytes"] for x in pk),
                                  "dropped": len(pd_), "dropped_bytes": sum(x["bytes"] for x in pd_),
                                  "kept_why": dict(Counter(x["why"] for x in pk)),
                                  "dropped_why": dict(Counter(x["why"] for x in pd_)),
                                  "coverage_missing": cover, "files": pk}
        RUN_FILES_ACCOUNT[run.new] = acc
        log(f"  {run.new:32s} <- {run.old:58s} probes kept {len(pk):3d} "
            f"({PROBE_ACCOUNT[run.new]['kept_bytes'] / 1e6:6.1f} MB) dropped {len(pd_):3d} "
            f"({PROBE_ACCOUNT[run.new]['dropped_bytes'] / 1e6:6.1f} MB)  [{time.time() - t0:.0f}s]")
        if cover:
            log(f"    !! coverage gaps: {cover}")
    return fps


def _torch_load_prov(p: Path) -> dict:
    import torch
    return torch.load(p, map_location="cpu", weights_only=False, mmap=True)["provenance"]


def probe_coverage(run: Run, scores: dict, slots: list) -> list:
    """Every kept block must have its forward LIN+MLP fits and (where the block carries IM arms) 9 IM points."""
    have = Counter()
    for sl in slots:
        kind, tgt, tgts, fam, fams, basis, point = sl[:7]
        have[(kind, tgt, tgts, fam, fams, basis)] += 1
    gaps = []
    if run.env == "othello":
        if have[("othello_grid", None, "['mine']", None, "['linear', 'mlp']", None)] != 1:
            gaps.append("grid mine")
        if have[("inverse_map", "mine-onehot", "None", None, "None", None)] != scores["n_points"]:
            gaps.append("IM mine-onehot points")
        return gaps
    for key, b in scores["bases"].items():
        tgt, basis = b["target"], b["basis"]
        for fam in ("linear", "mlp"):
            if have[(None, tgt, "None", fam, "None", basis)] != 1:
                gaps.append(f"forward {tgt}/{basis}/{fam}")
        if any(a.get("editor") == "IM" for a in b.get("arms", [])) and tgt != "pos@appearance":
            if have[("inverse_map", tgt, "None", None, "None", basis)] != scores["n_points"]:
                gaps.append(f"IM {tgt}/{basis} points {have[('inverse_map', tgt, 'None', None, 'None', basis)]}")
    return gaps


def du(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                t += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return t


# ── step: baselines ──────────────────────────────────────────────────────────────────────────────────

BASELINE_ACCOUNT: dict = {}


def baseline_scope(env_old: str, inst_old: str, runs: list[Run]) -> dict:
    """{arch: kept bases/targets} for one instance, from the in-scope runs trained on it."""
    out: dict[str, set] = {}
    for r in runs:
        if r.env_old == env_old and r.inst_old == inst_old and not r.replicate:
            keep = {"mine/theirs"} if r.env == "othello" else (set(r.bases) | set(r.extras))
            out.setdefault(r.arch, set()).update(keep)
    return out


def export_baselines(runs: list[Run]):
    from pim.probes.baselines import random_init_model
    from pim.probes.cache import fingerprint
    from pim.models import load_checkpoint
    for (env_old, inst_old), inst_new in INSTANCES.items():
        env = ENV_NEW[env_old]
        src = P_RUNS / "_baselines" / inst_old
        rd = f"runs/_baselines/{env}/{inst_new}"
        scope = baseline_scope(env_old, inst_old, runs)
        # random-init fingerprints of the kept architectures (seeded CPU init, as the scorer builds them)
        rfp, span, toktag = {}, {}, {}
        for arch in scope:
            r0 = next(r for r in runs if r.env_old == env_old and r.inst_old == inst_old and r.arch == arch
                      and not r.replicate)
            _m, info = load_checkpoint(r0.src / "best_model.pt", device="cpu")
            assert info.arch == arch, (r0.old, info.arch, arch)
            rm = random_init_model(info.arch, info.model_config, seed=0, device="cpu")
            rfp[arch], span[arch] = fingerprint(rm), int(getattr(rm, "state_span", -1))
            toktag[arch] = tokens_tag(r0)
            del _m, rm
        # baselines.json
        bj = read_json(src / "baselines.json")
        b = copy.deepcopy(bj)
        b["archs"] = {a: {**v, "bases": {k: blk for k, blk in v["bases"].items() if k in scope[a]}}
                      for a, v in b["archs"].items() if a in scope}
        b.pop("commit_sha", None)
        drop_keys(b)
        exp = copy.deepcopy(b)
        new = string_pass(b)
        new["instance"] = inst_new
        if "env" in new:
            new["env"] = env
        put_bytes(f"{rd}/baselines.json", dump_json(new), private_rel(src / "baselines.json"), "transformed")
        record_number_check(f"{rd}/baselines.json", exp, new, private_rel(src / "baselines.json"))
        absent = {a: sorted(scope[a] - set(new["archs"].get(a, {}).get("bases", {}))) for a in scope}
        # bayes_floor.json (canonical), reachability.json, corpus_stats.json
        for name in ("bayes_floor.json", "reachability.json", "corpus_stats.json"):
            if (src / name).exists():
                n2, e2 = transform_generic(read_json(src / name))
                if "instance" in n2:
                    n2["instance"] = inst_new
                put_bytes(f"{rd}/{name}", dump_json(n2), private_rel(src / name), "transformed")
                record_number_check(f"{rd}/{name}", e2, n2, private_rel(src / name))
        dropped_files = [p.name for p in sorted(src.iterdir())
                         if p.is_file() and p.name not in ("baselines.json", "bayes_floor.json", "reachability.json",
                                                           "corpus_stats.json")]
        # probes
        targets_any = set().union(*scope.values()) if scope else set()
        kept, dropped = [], []
        slots = {}
        for p in sorted((src / "probes").iterdir()):
            if p.is_dir() or not p.name.startswith("probes_") or p.suffix != ".pt":
                dropped.append({"file": p.name, "bytes": du(p), "why": "INDEX.md / not a cache file"})
                continue
            prov = _torch_load_prov(p)
            keep, why = classify_baseline_probe(prov, env_old, inst_old, scope, rfp, span, toktag, targets_any)
            if not keep:
                dropped.append({"file": p.name, "bytes": p.stat().st_size, "why": why})
                continue
            sl = probe_slot(prov) + (prov.get("span"), prov.get("encoder"), prov.get("split"))
            assert sl not in slots, f"duplicate baseline slot {inst_old}: {slots[sl]} {p.name}"
            slots[sl] = p.name
            info = ship_probe(p, f"{rd}/probes", rekey=(env == "rayworld"))
            info["why"] = why
            kept.append(info)
        BASELINE_ACCOUNT[f"{env}/{inst_new}"] = {
            "old": inst_old, "archs": {a: sorted(v) for a, v in scope.items()}, "random_init_fp": rfp,
            "archs_dropped": sorted(set(bj["archs"]) - set(scope)),
            "blocks_dropped": {a: sorted(set(v["bases"]) - scope.get(a, set())) for a, v in bj["archs"].items()},
            "in_scope_but_absent": absent, "files_dropped": dropped_files,
            "probes_kept": len(kept), "probes_kept_bytes": sum(x["bytes"] for x in kept),
            "probes_dropped": len(dropped), "probes_dropped_bytes": sum(x["bytes"] for x in dropped),
            "kept_why": dict(Counter(x["why"] for x in kept)), "dropped_why": dict(Counter(x["why"] for x in dropped))}
        log(f"  {rd:40s} probes kept {len(kept):3d} ({BASELINE_ACCOUNT[f'{env}/{inst_new}']['probes_kept_bytes'] / 1e6:6.1f} MB)"
            f" dropped {len(dropped):3d}  archs {sorted(scope)}")


def classify_baseline_probe(prov, env_old, inst_old, scope, rfp, span, toktag, targets_any) -> tuple[bool, str]:
    if prov.get("v") != 2 or int(prov.get("seed", 0)) != 0:
        return False, "cache version / seed"
    model = prov.get("model")
    if env_old == "othello":
        if model == "none" and prov.get("kind") == "othello_observation":
            return (True, "observation mine") if prov.get("target") == "mine" else (False, f"observation {prov.get('target')}")
        arch = next((a for a, f in rfp.items() if f == model), None)
        if arch and prov.get("kind") == "othello_grid":
            return (True, "random-init grid mine") if prov.get("targets") == ["mine"] else \
                (False, f"random-init grid {prov.get('targets')}")
        return False, f"model {model} not a kept architecture"
    if prov.get("data") != f"discworld/{inst_old}":
        return False, f"data {str(prov.get('data'))[:40]!r} (legacy path-keyed)" if str(prov.get("data")).startswith("/") \
            else f"data {prov.get('data')!r}"
    bases = ("cartesian",) if inst_old == "dw-8ray-obs5" else ("frustum", "cartesian")
    tgt, basis = prov.get("target"), prov.get("basis")

    def target_ok(allowed: set) -> bool:
        if tgt == "full":
            return basis in bases and basis in allowed
        return tgt in allowed and basis == bases[0]
    if model == "none":
        if prov.get("kind") != "observation" or prov.get("span") not in span.values():
            return False, f"observation span {prov.get('span')} (not a kept architecture)"
        return (True, f"observation {tgt}/{basis}") if target_ok(targets_any) else (False, f"observation {tgt} out of scope")
    arch = next((a for a, f in rfp.items() if f == model), None)
    if arch is None:
        return False, f"random-init fp {model} (dropped architecture)"
    if prov.get("encoder") != toktag[arch]:
        return False, "encoder mismatch"
    return (True, f"random-init[{arch}] {tgt}/{basis}") if target_ok(scope[arch]) else \
        (False, f"random-init[{arch}] {tgt} out of scope")


# ── step: datasets ───────────────────────────────────────────────────────────────────────────────────

DATASET_ACCOUNT: dict = {}


OTHELLO_SPLIT_HELPER = r'''
import base64, io, json, sys
import numpy as np
import pim
assert pim.__file__.startswith(sys.argv[1] + "/"), pim.__file__
from pim.environments.othello import corpus as oc


class Buffer(io.BytesIO):
    """The split file: np.savez writes into memory (it takes any object with write/read)."""
    def exists(self):
        return False

    def stat(self):
        return type("Stat", (), {"st_size": len(self.getvalue())})()


class Dir:
    """An empty split directory that is never created; its one file is a Buffer."""
    file = None

    def glob(self, pat):
        return iter(())

    def mkdir(self, **kw):
        pass

    def __truediv__(self, name):
        self.file = Buffer()
        self.file.name = name
        return self.file


out = []
for req in json.loads(sys.stdin.read()):
    z = np.load(req["src"])
    tok, ln, seen, d = z["tokens"], z["lengths"], {}, Dir()

    def generate(lo, n, log=None, flip=True, placement="enclosure", **kw):
        assert (lo, n) == (int(z["lo"]), len(tok)), (lo, n)
        seen.update(lo=lo, n=n, flip=flip, placement=placement)
        return tok, ln

    oc._generate, oc.corpus_dir = generate, (lambda instance, split: d)
    got = oc.build(only=(req["split"],), instance=req["instance"], log=lambda s: None)[req["split"]]
    assert got is d.file and got.name == req["name"], (got, req)
    out.append({**req, "rules": seen, "bytes": base64.b64encode(got.getvalue()).decode()})
print(json.dumps(out))
'''


def release_othello_splits(reqs: list[dict]) -> dict:
    """{rel: bytes} of each Othello split as the RELEASE corpus.build writes it, the source file's games standing in
    for generation (RELEASE subprocess; writes nothing). Each req: src, split, name, instance, rel."""
    import base64
    env = dict(os.environ, PYTHONPATH=str(RELEASE), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(PY), "-c", OTHELLO_SPLIT_HELPER, str(RELEASE)], input=json.dumps(reqs),
                       cwd=RELEASE, env=env, capture_output=True, text=True, check=True)
    out = {}
    for x in json.loads(r.stdout):
        out[x["rel"]] = (base64.b64decode(x["bytes"]), x["rules"])
    return out


def othello_split_check(src: Path, data: bytes, inst_old: str, inst_new: str, rules: dict) -> dict:
    """The rendered split holds the source's arrays bit for bit (every key but `instance`), the rules the source was
    generated under (PRIVATE corpus rules of its old instance) and the new instance name."""
    import numpy as np
    from pim.environments.othello import corpus as poc
    want = {"flip": poc.flip_of(inst_old), "placement": poc.placement_of(inst_old)}
    z, back = np.load(src), np.load(io.BytesIO(data))
    assert {k: rules[k] for k in want} == want, (src, rules, want)
    assert set(z.files) <= set(back.files) and set(back.files) - set(z.files) <= {"flip", "placement", "instance"}, \
        (z.files, back.files)
    for k in z.files:
        if k != "instance":
            a, b = z[k], back[k]
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (src, k)
    assert bool(back["flip"]) == want["flip"] and str(back["placement"]) == want["placement"], src
    assert str(back["instance"]) == inst_new and back["instance"].dtype == np.array(inst_new).dtype, src
    assert (int(back["lo"]), len(back["tokens"])) == (rules["lo"], rules["n"]), src
    return {"keys": back.files, "added": [k for k in back.files if k not in z.files],
            "instance": [str(z["instance"]) if "instance" in z.files else None, inst_new]}


OTHELLO_SPLIT_ACCOUNT: dict = {}


def export_othello_splits(env_old: str, inst_old: str, inst_new: str):
    src_dir, dd = P_DATA / env_old / inst_old, f"datasets/othello/{inst_new}"
    reqs = [{"src": str(src_dir / sub), "split": split, "name": Path(sub).name, "instance": inst_new,
             "rel": f"{dd}/{sub}"} for split, sub in OTHELLO_SPLITS]
    rendered = release_othello_splits(reqs)
    for q in reqs:
        data, rules = rendered[q["rel"]]
        OTHELLO_SPLIT_ACCOUNT[q["rel"]] = othello_split_check(Path(q["src"]), data, inst_old, inst_new, rules)
        put_bytes(q["rel"], data, private_rel(Path(q["src"])),
                  "npz rendered by the release corpus.build (source games; release keys and rules)")


def transform_dataset_json(old: dict, rel: str) -> tuple[dict, dict]:
    """A dataset manifest: generic scrub, no `layout` key, the new instance name; an Othello cases manifest
    gets the American-spelled recipe (no code reads it)."""
    d = copy.deepcopy(old)
    d.pop("layout", None)
    new, expected = transform_generic(d)
    if "instance" in new:
        new["instance"] = rel.split("/")[2]
    if rel.startswith("datasets/othello/") and "/edits/cases_" in rel:
        assert new["recipe"] == OTHELLO_CASES_RECIPE_OLD, new["recipe"]
        new["recipe"] = OTHELLO_CASES_RECIPE
    return new, expected


# ── HDF5: a fresh file without config_json's generated_at (an in-place attribute rewrite keeps the old bytes) ──

H5_RECORD_P = WORK / "h5_scrubbed.json"
H5_RECORD: dict = {}


def _dset_props(ds) -> tuple:
    p = ds.id.get_create_plist()
    fill = ds.fillvalue.tobytes() if hasattr(ds.fillvalue, "tobytes") else ds.fillvalue
    return (ds.dtype.str, ds.shape, ds.maxshape, ds.chunks, ds.compression, ds.compression_opts, ds.shuffle,
            ds.fletcher32, ds.scaleoffset, fill, p.get_fill_time(), p.get_alloc_time(),
            [p.get_filter(i) for i in range(p.get_nfilters())], sorted(ds.attrs.keys()))


def h5_config(raw: str) -> tuple[str, str]:
    """(config_json without generated_at, the removed value); the stored text is json.dumps(indent=2)."""
    c = json.loads(raw)
    assert json.dumps(c, indent=2) == raw, "config_json is not in the json.dumps(indent=2) form"
    gen = c.pop(H5_DROP)
    return json.dumps(c, indent=2), gen


def h5_scrub(src: Path, dst: Path) -> str:
    """Copy every dataset (H5Ocopy: raw chunks, filters, fill value) and root attribute of `src` into a new file
    `dst`; config_json loses generated_at. Returns the removed value."""
    import h5py
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        for k in s:
            assert isinstance(s[k], h5py.Dataset), (src, k)
            s.copy(s[k], d, name=k)
        gen = None
        for k, v in s.attrs.items():
            if k == "config_json":
                d.attrs[k], gen = h5_config(v)
            else:
                d.attrs[k] = v
    assert gen is not None, src
    return gen


def h5_compare(src: Path, new: Path, full: bool) -> dict:
    """`new` holds `src`'s datasets bit for bit (properties, filters, raw chunk bytes; with `full` also the decoded
    arrays) and its root attributes, config_json without generated_at."""
    import h5py
    out = {"datasets": 0, "chunks": 0, "decoded_bytes": 0}
    with h5py.File(src, "r") as s, h5py.File(new, "r") as d:
        assert list(s.keys()) == list(d.keys()), (list(s), list(d))
        assert list(s.attrs.keys()) == list(d.attrs.keys())
        for k in s.attrs:
            ta, tb = h5py.h5a.open(s.id, k.encode()).get_type(), h5py.h5a.open(d.id, k.encode()).get_type()
            assert type(ta) is type(tb), k
            if k == "config_json":
                assert d.attrs[k] == h5_config(s.attrs[k])[0]
                assert ta.is_variable_str() and tb.is_variable_str() and ta.get_cset() == tb.get_cset()
            else:
                assert s.attrs[k] == d.attrs[k], k
        for k in s:
            a, b = s[k], d[k]
            assert _dset_props(a) == _dset_props(b), (k, _dset_props(a), _dset_props(b))
            assert a.chunks is not None, (src, k)
            n = a.id.get_num_chunks()
            assert n == b.id.get_num_chunks(), k
            for i in range(n):
                ia, ib = a.id.get_chunk_info(i), b.id.get_chunk_info(i)
                assert (ia.chunk_offset, ia.filter_mask, ia.size) == (ib.chunk_offset, ib.filter_mask, ib.size), (k, i)
                assert a.id.read_direct_chunk(ia.chunk_offset) == b.id.read_direct_chunk(ib.chunk_offset), (k, i)
            out["chunks"] += n
            if full:
                step = max(1, (64 << 20) // max(1, a.dtype.itemsize * int(math.prod(a.shape[1:]) or 1)))
                for lo in range(0, a.shape[0], step):
                    xa, xb = a[lo: lo + step], b[lo: lo + step]
                    assert xa.dtype == xb.dtype and xa.shape == xb.shape and xa.tobytes() == xb.tobytes(), (k, lo)
                    out["decoded_bytes"] += xa.nbytes
            out["datasets"] += 1
    return out


def put_h5(rel: str, src: Path):
    d = _dest(rel)
    src_sha = sha256_file(src)
    rec = H5_RECORD.get(rel)
    if not (rec and rec["src_sha256"] == src_sha and d.exists() and not d.is_symlink()
            and sha256_file(d) == rec["sha256"]):
        tmp = d.with_name(d.name + ".partial")
        h5_scrub(src, tmp)
        check = h5_compare(src, tmp, full=True)
        os.replace(tmp, d)
        H5_RECORD[rel] = {"src": private_rel(src), "src_sha256": src_sha, "sha256": sha256_file(d),
                          "src_bytes": src.stat().st_size, "bytes": d.stat().st_size, "check": check}
        log(f"    {rel}: rewritten without {H5_DROP} ({check['datasets']} datasets, {check['chunks']} chunks, "
            f"{check['decoded_bytes'] / 1e9:.2f} GB decoded, all identical)")
    WRITTEN[rel] = {"src": private_rel(src), "how": f"h5 rewritten (config_json without {H5_DROP})"}


# ── generated artifacts and probe indexes ────────────────────────────────────────────────────────────

def export_generated():
    for root, _, files in os.walk(GENERATED):
        for f in sorted(files):
            p = Path(root) / f
            put_copy(str(p.relative_to(GENERATED)), p, "copy (generated)")


INDEX_HELPER = r'''
import json, sys
from pathlib import Path
import pim
assert pim.__file__.startswith(sys.argv[1] + "/"), pim.__file__
from pim.probes.cache import ProbeCache


class Capture:
    """A cache directory that globs the real one and captures INDEX.md instead of writing it."""
    def __init__(self, d):
        self.d, self.text = d, None

    def glob(self, pat):
        return self.d.glob(pat)

    def __truediv__(self, name):
        assert name == "INDEX.md", name
        cap = self

        class Out:
            def write_text(self, t):
                cap.text = t
        return Out()


out = {}
for rel in json.loads(sys.stdin.read()):
    c = ProbeCache(Path(sys.argv[2]) / rel)
    c.dir = Capture(c.dir)
    c.write_index()
    out[rel] = c.dir.text
print(json.dumps(out))
'''


def release_index_texts(dirs: list[str]) -> dict:
    """{probes dir: INDEX.md text} rendered by the RELEASE ProbeCache.write_index (RELEASE subprocess, reads only)."""
    env = dict(os.environ, PYTHONPATH=str(RELEASE), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(PY), "-c", INDEX_HELPER, str(RELEASE), str(STAGING)], input=json.dumps(dirs),
                       cwd=RELEASE, env=env, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def probe_dirs() -> list[str]:
    return sorted({str(Path(rel).parent) for rel in WRITTEN
                   if Path(rel).parent.name == "probes" and Path(rel).name.startswith("probes_")})


def export_probe_indexes():
    texts = release_index_texts(probe_dirs())
    for rel_dir, text in texts.items():
        put_bytes(f"{rel_dir}/INDEX.md", text.encode("utf-8"), "release pim.probes.cache.ProbeCache.write_index",
                  "derived (release ProbeCache.write_index)")
    log(f"  {len(texts)} probes/INDEX.md rendered by the release ProbeCache.write_index")


def export_datasets():
    for (env_old, inst_old), inst_new in INSTANCES.items():
        env = ENV_NEW[env_old]
        src = P_DATA / env_old / inst_old
        dd = f"datasets/{env}/{inst_new}"
        files = []
        if env == "othello":
            export_othello_splits(env_old, inst_old, inst_new)
            put_copy(f"{dd}/edits/cases_1000.pkl", src / "edits" / "v1" / "cases_1000.pkl")
            jsons = [(src / "edits" / "v1" / "cases_1000.json", f"{dd}/edits/cases_1000.json")]
        else:
            put_h5(f"{dd}/eval/test.h5", src / "eval" / "test.h5")
            put_h5(f"{dd}/edits/edits.h5", src / "edits" / "v1" / "edits.h5")
            put_h5(f"{dd}/probe/probe_120k.h5", src / "probe" / "probe_120k.h5")
            jsons = [(src / "eval" / "test.json", f"{dd}/eval/test.json"),
                     (src / "edits" / "v1" / "edits.json", f"{dd}/edits/edits.json"),
                     (src / "edits" / "v1" / "selection.json", f"{dd}/edits/selection.json"),
                     (src / "probe" / "probe_120k.json", f"{dd}/probe/probe_120k.json")]
            if inst_new in RAY_FAMILY:
                put_h5(f"{dd}/probe/probe_250k.h5", src / "probe" / "probe_250k.h5")
                jsons.append((src / "probe" / "probe_250k.json", f"{dd}/probe/probe_250k.json"))
            if inst_new == "8-ray":
                put_copy(f"{dd}/tokens/vocab.npz", src / "tokens" / "vocab.npz")
        for s, rel in jsons:
            n2, e2 = transform_dataset_json(read_json(s), rel)
            put_bytes(rel, dump_json(n2), private_rel(s), "transformed")
            record_number_check(rel, e2, n2, private_rel(s))
        shipped_src = {v["src"] for k, v in WRITTEN.items() if k.startswith(dd + "/")}
        not_shipped = []
        for root, dirs, fs in os.walk(src):
            for f in fs:
                p = Path(root) / f
                if private_rel(p) not in shipped_src:
                    not_shipped.append((str(p.relative_to(src)), p.stat().st_size))
        DATASET_ACCOUNT[f"{env}/{inst_new}"] = {
            "old": f"{env_old}/{inst_old}",
            "shipped": sorted(k[len(dd) + 1:] for k in WRITTEN if k.startswith(dd + "/")),
            "not_shipped_top": dict(Counter(n.split("/")[0] for n, _ in not_shipped)),
            "not_shipped_bytes": sum(b for _, b in not_shipped)}
        log(f"  {dd:34s} shipped {len(DATASET_ACCOUNT[f'{env}/{inst_new}']['shipped'])} files")


# ── step: manifest ───────────────────────────────────────────────────────────────────────────────────

def bundle_of(rel: str) -> str:
    parts = rel.split("/")
    if parts[0] == "datasets":
        return "corpora" if len(parts) > 3 and parts[3] == "probe" else "core"
    if parts[0] == "runs" and parts[1] != "_baselines" and "__seed" in parts[2] \
            and parts[3] in ("best_model.pt", "probes"):
        return "replicates"
    return "core"


def all_staging_files() -> list[str]:
    out = []
    for root, dirs, files in os.walk(STAGING):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(STAGING))
            if rel in ("MANIFEST.json", "SHA256SUMS"):
                continue
            out.append(rel)
    return sorted(out)


def write_manifest():
    files = []
    sizes = Counter()
    counts = Counter()
    for rel in all_staging_files():
        p = STAGING / rel
        b = bundle_of(rel)
        files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_file(p), "bundle": b})
        sizes[b] += p.stat().st_size
        counts[b] += 1
    man = {"description": "Artifact bundle for the release: trained models, fitted probes, scores, baselines and "
                          "the evaluation, edit-bench and probe datasets. Paths are relative to the repository root.",
           "bundles": {b: {"files": counts[b], "bytes": sizes[b]} for b in ("core", "corpora", "replicates")},
           "files": files}
    (STAGING / "MANIFEST.json").write_bytes(dump_json(man))
    (STAGING / "SHA256SUMS").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in files))
    for b in ("core", "corpora", "replicates"):
        log(f"  bundle {b:10s} {counts[b]:5d} files  {sizes[b] / 1e9:8.3f} GB")
    return man


# ── step: verify ─────────────────────────────────────────────────────────────────────────────────────

def verify(runs: list[Run]) -> dict:
    res = {}
    # (a) re-key: every shipped probe file's stored provenance hashes to its filename; data field is new-style
    n_ok, bad, data_vals = 0, [], Counter()
    for rel in all_staging_files():
        if "/probes/" in rel and rel.endswith(".pt"):
            prov = _torch_load_prov(STAGING / rel)
            if cache_fname(prov) != Path(rel).name:
                bad.append(rel)
            else:
                n_ok += 1
            if "data" in prov:
                data_vals[prov["data"]] += 1
    res["rekey"] = {"files": n_ok + len(bad), "ok": n_ok, "bad": bad, "data_values": dict(data_vals)}
    log(f"  (a) re-key: {n_ok}/{n_ok + len(bad)} probe files hash to their name")
    # (b) numbers: re-derive every transformed JSON from its source and compare with the file on disk
    run_by_new = {r.new: r for r in runs}
    nb, nfail, nleaves = 0, [], 0
    for rel in all_staging_files():
        if not rel.endswith(".json") or rel in ("MANIFEST.json",):
            continue
        w = WRITTEN.get(rel)
        if w is None:
            continue
        srcp = PRIVATE / w["src"] if not w["src"].startswith("/") else Path(w["src"])
        new = read_json(STAGING / rel)
        old = read_json(srcp)
        parts = rel.split("/")
        if rel.startswith("runs/") and parts[1] != "_baselines" and parts[3] == "scores.json":
            _, exp = transform_scores(old, run_by_new[f"{parts[1]}/{parts[2]}"])
        elif rel.startswith("runs/") and parts[1] != "_baselines" and parts[3] == "config.json":
            _, exp = transform_config(old, run_by_new[f"{parts[1]}/{parts[2]}"])
        elif parts[-1] == "baselines.json":
            env, inst = parts[2], parts[3]
            inst_old = next(i for (e, i), n in INSTANCES.items() if ENV_NEW[e] == env and n == inst)
            scope = baseline_scope("othello" if env == "othello" else "discworld", inst_old, runs)
            b = copy.deepcopy(old)
            b["archs"] = {a: {**v, "bases": {k: blk for k, blk in v["bases"].items() if k in scope[a]}}
                          for a, v in b["archs"].items() if a in scope}
            b.pop("commit_sha", None)
            drop_keys(b)
            exp = b
        elif rel.startswith("datasets/"):
            _, exp = transform_dataset_json(old, rel)
        else:
            _, exp = transform_generic(old)
        r = compare_numbers(exp, new)
        nb += 1
        nleaves += r["n"]
        if not r["ok"]:
            nfail.append({"file": rel, **r})
    res["numbers"] = {"files": nb, "numeric_leaves": nleaves, "failures": nfail}
    log(f"  (b) numbers: {nb} JSON files, {nleaves} numeric leaves, {len(nfail)} failures")
    # (c) identity scan
    res["identity"] = identity_scan()
    log(f"  (c) identity scan: {len(res['identity']['hits'])} hits (dates included, HDF5 attrs included); "
        f"raw-byte scan of {res['identity']['raw_scan']['files']} files "
        f"({res['identity']['raw_scan']['bytes'] / 1e9:.2f} GB): {len(res['identity']['raw_scan']['hits'])} hits")
    res["h5"] = verify_h5()
    log(f"  HDF5: {res['h5']['files']} files match their source (datasets, raw chunks, attrs); "
        f"old timestamp strings found: {len(res['h5']['old_timestamp_hits'])}; bad {len(res['h5']['bad'])}")
    res["release_form"] = verify_release_form(runs)
    log(f"  release form: {json.dumps({k: v for k, v in res['release_form'].items() if k != 'bench_selection'})}")
    log(f"  bench_selection: {json.dumps(res['release_form']['bench_selection'])}")
    res["othello_splits"] = verify_othello_splits()
    log(f"  Othello splits: {res['othello_splits']['files']} equal the release corpus.build rendering, hold the "
        f"source arrays and rules, pass the release verify_splits; bad {res['othello_splits']['bad']}")
    res["probe_index"] = verify_probe_indexes()
    log(f"  probes/INDEX.md: {res['probe_index']['dirs']} dirs equal the release write_index output, "
        f"rows cover exactly the dir's files; bad {res['probe_index']['bad']}")
    # checkpoints: fingerprint of every shipped model, via the model loader, matches the source run
    fpbad = []
    for r in runs:
        a = model_fingerprint(STAGING / "runs" / r.new / "best_model.pt")
        b = PROBE_ACCOUNT.get(r.new, {}).get("fingerprint") or state_fingerprint(
            _torch_load(r.src / "best_model.pt")["model_state"])
        if a != b:
            fpbad.append((r.new, a, b))
    res["checkpoint_fingerprints"] = {"checked": len(runs), "mismatch": fpbad}
    log(f"  checkpoint fingerprints: {len(runs) - len(fpbad)}/{len(runs)} match")
    # verbatim copies are byte-identical
    cbad = [rel for rel, w in WRITTEN.items() if w["how"].startswith("copy")
            and sha256_file(STAGING / rel) != sha256_file(PRIVATE / w["src"])]
    res["verbatim_copies"] = {"checked": sum(1 for w in WRITTEN.values() if w["how"].startswith("copy")), "bad": cbad}
    # no symlinks / hardlinks
    links = [rel for rel in all_staging_files() if (STAGING / rel).is_symlink() or (STAGING / rel).stat().st_nlink > 1]
    res["links"] = links
    return res


def _pickle_strings(data: bytes) -> tuple[list, list]:
    strs, glob, prev = [], [], []
    for op, arg, pos in pickletools.genops(io.BytesIO(data)):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING",
                       "BINSTRING", "STRING"):
            s = arg if isinstance(arg, str) else str(arg)
            strs.append(s)
            prev.append(s)
        elif op.name == "GLOBAL":
            glob.append(str(arg).replace(" ", "."))
        elif op.name == "STACK_GLOBAL" and len(prev) >= 2:
            glob.append(prev[-2] + "." + prev[-1])
    return strs, glob


GLOBALS_SEEN: Counter = Counter()


def identity_scan() -> dict:
    import h5py
    import numpy as np
    hits, gen_at, scanned = [], [], Counter()
    low_ids = IDENTITY

    def check(where: str, text: str, dates: bool):
        low = text.lower()
        for t in low_ids:
            if t in low:
                i = low.index(t)
                hits.append({"where": where, "token": t, "context": text[max(0, i - 40): i + 60]})
        if dates and DATE_RE.search(text):
            m = DATE_RE.search(text)
            hits.append({"where": where, "token": "date", "context": text[max(0, m.start() - 40): m.end() + 40]})

    for root, dirs, files in os.walk(STAGING):
        for n in dirs + files:
            rel = str((Path(root) / n).relative_to(STAGING))
            check(f"name:{rel}", rel, True)
    for rel in all_staging_files() + ["MANIFEST.json", "SHA256SUMS"]:
        p = STAGING / rel
        if not p.exists():
            continue
        suf = p.suffix
        if suf in (".json", ".jsonl", ".md", ".txt") or p.name == "SHA256SUMS":
            check(rel, p.read_text(), True)
            scanned["text"] += 1
        elif suf == ".pt":
            with zipfile.ZipFile(p) as z:
                names = z.namelist()
                pkl = [x for x in names if x.endswith("data.pkl")][0]
                strs, glob = _pickle_strings(z.read(pkl))
                for x in names:
                    check(f"{rel}::zipname", x, True)
            for g in glob:
                GLOBALS_SEEN[g] += 1
            for s in strs:
                check(f"{rel}::pickle", s, True)
            scanned["pickle"] += 1
        elif suf == ".pkl":
            strs, glob = _pickle_strings(p.read_bytes())
            for s in strs:
                check(f"{rel}::pickle", s, True)
            for g in glob:
                GLOBALS_SEEN[g] += 1
            scanned["pkl"] += 1
        elif suf == ".npz":
            z = np.load(p, allow_pickle=False)
            for k in z.files:
                check(f"{rel}::key", k, True)
                a = z[k]
                if a.dtype.kind in "SU":
                    for s in np.atleast_1d(a).ravel().tolist():
                        check(f"{rel}::{k}", str(s), True)
            scanned["npz"] += 1
        elif suf == ".h5":
            with h5py.File(p, "r") as h:
                def visit_attrs(name, obj):
                    for k, v in obj.attrs.items():
                        s = v.decode() if isinstance(v, bytes) else str(v)
                        if H5_DROP in s:
                            gen_at.append(f"{rel}:{name or '/'}@{k}")
                        check(f"{rel}:{name or '/'}@{k}", s, True)
                    if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "SOU":
                        for s in np.atleast_1d(obj[()]).ravel().tolist()[:10000]:
                            check(f"{rel}:{name}", s.decode() if isinstance(s, bytes) else str(s), True)
                visit_attrs("", h)
                h.visititems(visit_attrs)
            scanned["h5"] += 1
        else:
            scanned[f"other{suf}"] += 1
    raw_hits, raw_bytes, raw_files = [], 0, 0
    for rel in all_staging_files() + ["MANIFEST.json", "SHA256SUMS"]:
        p = STAGING / rel
        if p.exists():
            raw_hits += [{"where": rel, "offset": off, "token": tok} for tok, off in raw_scan(p, RAW_TOKENS, RAW_DATE)]
            raw_bytes += p.stat().st_size
            raw_files += 1
    return {"hits": hits, "hdf5_generated_at": gen_at, "scanned": dict(scanned),
            "pickle_globals": dict(GLOBALS_SEEN),
            "raw_scan": {"files": raw_files, "bytes": raw_bytes, "hits": raw_hits[:200],
                         "tokens": [t.decode() for t in RAW_TOKENS], "date_pattern": RAW_DATE.pattern.decode()}}


def raw_scan(p: Path, tokens: list, date_re=None, block: int = 64 << 20) -> list:
    """(token, byte offset) of every occurrence of a token (or a date match) in the raw bytes of `p`."""
    hits, pad = [], max(max((len(t) for t in tokens), default=0), 16)
    with open(p, "rb") as f:
        base, tail = 0, b""
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            buf = tail + chunk
            start = base - len(tail)
            for t in tokens:
                i = buf.find(t)
                while i != -1:
                    if i + len(t) > len(tail):          # not already reported from the previous block
                        hits.append((t.decode(), start + i))
                    i = buf.find(t, i + 1)
            if date_re is not None:
                for m in date_re.finditer(buf):
                    if m.end() > len(tail):
                        hits.append(("date:" + m.group(0).decode(), start + m.start()))
            tail = buf[-pad:]
            base += len(chunk)
    return hits


def verify_h5() -> dict:
    """Every shipped .h5 against its PRIVATE source (h5_compare: properties, filters, raw chunk bytes, attrs) and its
    raw bytes against the generated_at value the source stores."""
    import h5py
    files, bad, hits = 0, [], []
    for rel, w in sorted(WRITTEN.items()):
        if not rel.endswith(".h5"):
            continue
        src = PRIVATE / w["src"]
        try:
            h5_compare(src, STAGING / rel, full=False)
        except AssertionError as e:
            bad.append({"file": rel, "error": str(e)[:300]})
        with h5py.File(src, "r") as s:
            old = json.loads(s.attrs["config_json"])[H5_DROP]
        found = raw_scan(STAGING / rel, [old.encode(), H5_DROP.encode()])
        if found:
            hits.append({"file": rel, "hits": found[:5]})
        files += 1
    rec_ok = sum(1 for rel in H5_RECORD if rel in WRITTEN and H5_RECORD[rel]["sha256"] == sha256_file(STAGING / rel))
    return {"files": files, "bad": bad, "old_timestamp_hits": hits,
            "full_decode_checks_recorded": rec_ok,
            "decoded_bytes_recorded": sum(r["check"]["decoded_bytes"] for r in H5_RECORD.values())}


def _walk_json(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}/{k}", k, v
            yield from _walk_json(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield f"{path}[{i}]", None, v
            yield from _walk_json(v, f"{path}[{i}]")


def verify_release_form(runs: list[Run]) -> dict:
    """The release form of the shipped JSON and checkpoints: no `layout` keys or corpus/ paths, TrainConfig train
    blocks, the unread files gone, the generated files verbatim, and the bench_selection records."""
    import torch
    out = {"layout_keys": [], "corpus_paths": [], "train_blocks_bad": [], "legacy_keys_left": [],
           "ckpt_train_config_bad": [], "unread_present": [], "variance_shipped": [], "corpus_stats_shipped": [],
           "generated_bad": [], "legacy_runs": [], "replicate_notes": Counter(), "canonical_run_strings": []}
    assert f'"note": "{REPLICATE_NOTE}"' in (RELEASE / "scripts" / "make_replicate_member.py").read_text()
    for rel in all_staging_files():
        if rel.endswith(".json"):
            for path, k, v in _walk_json(read_json(STAGING / rel)):
                if k == "layout":
                    out["layout_keys"].append(f"{rel}:{path}")
                if isinstance(v, str) and "/corpus/" in v:
                    out["corpus_paths"].append(f"{rel}:{path}")
        if rel.endswith("/variance.json"):
            out["variance_shipped"].append(rel)
        if rel.endswith("/corpus_stats.json"):
            out["corpus_stats_shipped"].append(rel)
    for r in runs:
        cfg = read_json(STAGING / "runs" / r.new / "config.json")
        if "note" in cfg.get("replicate", {}):
            out["replicate_notes"][cfg["replicate"]["note"]] += 1
        if "canonical run" in (STAGING / "runs" / r.new / "config.json").read_text():
            out["canonical_run_strings"].append(r.new)
        if list(cfg["train"]) != list(TRAIN_FIELDS):
            out["train_blocks_bad"].append(r.new)
        left = [k for k in LEGACY_TOP_DROP if k in cfg] + [k for k in LEGACY_TRAIN_DROP if k in cfg["train"]]
        ck = torch.load(STAGING / "runs" / r.new / "best_model.pt", map_location="cpu", weights_only=False, mmap=True)
        left += [f"ckpt:{k}" for k in LEGACY_CKPT_DROP if k in ck]
        if not set(ck.get("train_config", {})) <= set(TRAIN_FIELDS) or "run_dir" in ck.get("train_config", {}):
            out["ckpt_train_config_bad"].append(r.new)
        if is_legacy_train(r.cfg["train"]):
            out["legacy_runs"].append(r.new)
            if not (cfg["train"] == LEGACY_TRAIN and ck["train_config"] == LEGACY_TRAIN):
                out["train_blocks_bad"].append(f"{r.new} (legacy values)")
        if left:
            out["legacy_keys_left"].append((r.new, left))
        del ck
    for run, name in sorted(UNREAD_RUN_FILES):
        if (STAGING / "runs" / run / name).exists():
            out["unread_present"].append(f"runs/{run}/{name}")
    for root, _, files in os.walk(GENERATED):
        for f in files:
            p = Path(root) / f
            q = STAGING / p.relative_to(GENERATED)
            if not q.exists() or sha256_file(q) != sha256_file(p):
                out["generated_bad"].append(str(p.relative_to(GENERATED)))
    nulls, filled = [], {}
    for r in runs:
        s = read_json(STAGING / "runs" / r.new / "scores.json")
        for k, b in s.get("bases", {}).items():
            if "bench_selection" in b and b["bench_selection"] is None:
                nulls.append(f"{r.new}:{k}")
        if r.new in BENCH_SELECTION_FILL:
            blk, sib = BENCH_SELECTION_FILL[r.new]
            filled[f"{r.new}:{blk}"] = s["bases"][blk]["bench_selection"] == s["bases"][sib]["bench_selection"]
    out["bench_selection"] = {"null_blocks": nulls, "filled_equal_sibling": filled}
    return out


SPLITS_VERIFY_HELPER = r'''
import json, sys
from pathlib import Path
import pim
assert pim.__file__.startswith(sys.argv[1] + "/"), pim.__file__
from pim.environments.othello import corpus as oc

out = {}
for inst, paths in json.loads(sys.stdin.read()).items():
    got = oc.verify_splits({k: Path(p) for k, p in paths.items()}, n_check=64, log=None)
    out[inst] = {k: list(v) for k, v in got.items()}
print(json.dumps(out))
'''


def verify_othello_splits() -> dict:
    """Each shipped Othello split equals the RELEASE corpus.build rendering of its source byte for byte, holds the
    source arrays and rules (othello_split_check), and passes the RELEASE verify_splits (disjoint index ranges;
    66 rows per split regenerated under the recorded rules)."""
    reqs, by_inst = [], {}
    for (env_old, inst_old), inst_new in INSTANCES.items():
        if env_old != "othello":
            continue
        dd = f"datasets/othello/{inst_new}"
        by_inst[inst_new] = {split: str(STAGING / dd / sub) for split, sub in OTHELLO_SPLITS}
        reqs += [{"src": str(P_DATA / env_old / inst_old / sub), "split": split, "name": Path(sub).name,
                  "instance": inst_new, "rel": f"{dd}/{sub}", "inst_old": inst_old} for split, sub in OTHELLO_SPLITS]
    rendered = release_othello_splits(reqs)
    bad = []
    for q in reqs:
        data, rules = rendered[q["rel"]]
        if (STAGING / q["rel"]).read_bytes() != data:
            bad.append(f"{q['rel']}: differs from the release corpus.build rendering")
        try:
            othello_split_check(Path(q["src"]), (STAGING / q["rel"]).read_bytes(), q["inst_old"], q["instance"], rules)
        except AssertionError as e:
            bad.append(f"{q['rel']}: {str(e)[:200]}")
    env = dict(os.environ, PYTHONPATH=str(RELEASE), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([str(PY), "-c", SPLITS_VERIFY_HELPER, str(RELEASE)], input=json.dumps(by_inst),
                       cwd=RELEASE, env=env, capture_output=True, text=True)
    if r.returncode:
        bad.append(f"release verify_splits: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    return {"files": len(reqs), "bad": bad, "release_verify_splits": json.loads(r.stdout) if not r.returncode else None,
            "sha256": {q["rel"]: sha256_file(STAGING / q["rel"]) for q in reqs}}


def verify_probe_indexes() -> dict:
    """Each probes/INDEX.md equals the RELEASE write_index output over the shipped dir and lists exactly its files."""
    dirs = sorted({str(Path(rel).parent) for rel in all_staging_files()
                   if Path(rel).parent.name == "probes" and Path(rel).name.startswith("probes_")})
    texts = release_index_texts(dirs)
    bad = []
    for d in dirs:
        p = STAGING / d / "INDEX.md"
        if not p.exists() or p.read_bytes() != texts[d].encode("utf-8"):
            bad.append(f"{d}: differs from the release write_index output")
            continue
        rows = sorted(re.findall(r"^\| `(probes_[0-9a-f]{16}\.pt)` \|", p.read_text(), flags=re.M))
        files = sorted(x.name for x in (STAGING / d).glob("probes_*.pt"))
        if rows != files:
            bad.append(f"{d}: rows {len(rows)} vs files {len(files)}")
    return {"dirs": len(dirs), "bad": bad}


# ── step: tables (reference for the table-diff gate) ─────────────────────────────────────────────────

def reference_tables():
    helper = WORK / "reference_tables.py"
    out = REPORTS / "reference_tables.json"
    env = dict(os.environ, PYTHONPATH=str(PRIVATE), MPLBACKEND="Agg")
    cmd = [str(PY), str(helper), "--out", str(out), "--shipped", str(WORK / "blocks_shipped.json")]
    log("  " + " ".join(cmd))
    subprocess.run(cmd, cwd=PRIVATE, env=env, check=True)


# ── driver ───────────────────────────────────────────────────────────────────────────────────────────

def make_writable():
    if STAGING.exists():
        for root, dirs, files in os.walk(STAGING):
            os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
            for f in files:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)


def lock():
    for root, dirs, files in os.walk(STAGING):
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for root, dirs, files in os.walk(STAGING, topdown=False):
        os.chmod(root, os.stat(root).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    log(f"  chmod -R a-w {STAGING}")


def prune_unplanned():
    """Move STAGING files this run did not write into REMOVED (a re-run after a scope change leaves no stale file;
    nothing is deleted). Empty directories are reported, not removed."""
    gone = []
    for rel in all_staging_files():
        if rel not in WRITTEN:
            dst, i = REMOVED / rel, 0
            while dst.exists():
                i += 1
                dst = REMOVED / f"{rel}.{i}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(STAGING / rel), str(dst))
            gone.append(rel)
    empty = [str(Path(root).relative_to(STAGING)) for root, dirs, files in os.walk(STAGING)
             if root != str(STAGING) and not dirs and not files]
    if empty:
        log(f"  empty directories left in STAGING: {empty}")
    return gone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "export", "manifest", "verify", "tables", "lock"])
    ap.add_argument("--force", action="store_true", help="make an existing (locked) STAGING writable first")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    load_sha_cache()
    if H5_RECORD_P.exists():
        H5_RECORD.update(read_json(H5_RECORD_P))
    runs = run_table()
    if STAGING.exists() and not os.access(STAGING, os.W_OK) and a.step in ("all", "export", "manifest"):
        if not a.force:
            sys.exit(f"{STAGING} is locked (read-only); pass --force to rebuild")
        make_writable()
    t0 = time.time()
    if a.step in ("all", "export"):
        STAGING.mkdir(parents=True, exist_ok=True)
        log("== runs")
        export_runs(runs)
        log("== baselines")
        export_baselines(runs)
        log("== datasets")
        export_datasets()
        H5_RECORD_P.write_text(json.dumps(H5_RECORD, indent=1, sort_keys=True))
        log("== generated")
        export_generated()
        log("== probe indexes")
        export_probe_indexes()
        gone = prune_unplanned()
        if gone:
            log(f"  moved {len(gone)} files no longer shipped into {REMOVED}: {gone}")
        (WORK / "written.json").write_text(json.dumps(WRITTEN, indent=1, sort_keys=True))
        (WORK / "blocks_shipped.json").write_text(json.dumps(BLOCKS_SHIPPED, indent=1))
        (WORK / "probe_accounting.json").write_text(json.dumps(
            {"runs": PROBE_ACCOUNT, "baselines": BASELINE_ACCOUNT, "run_files_dropped": RUN_FILES_ACCOUNT,
             "datasets": DATASET_ACCOUNT}, indent=1, default=str))
        (WORK / "number_checks.json").write_text(json.dumps(NUMBER_CHECKS, indent=1))
        (WORK / "othello_splits.json").write_text(json.dumps(OTHELLO_SPLIT_ACCOUNT, indent=1))
        vm = {"map": VERSION_MAP, "applied_counts": dict(VERSIONS_APPLIED),
              "fields": "every string under a key named 'version' or ending in '_version' (eval_version, "
                        "baseline_version, script_version, inverse_map.version, prediction.version, ...) "
                        "whose value is date-shaped; probe-cache provenance stores none"}
        (REPORTS / "version_map.json").write_text(json.dumps(vm, indent=1))
    else:
        if (WORK / "written.json").exists():
            WRITTEN.update(read_json(WORK / "written.json"))
        if (WORK / "probe_accounting.json").exists():
            PROBE_ACCOUNT.update(read_json(WORK / "probe_accounting.json")["runs"])
    if a.step in ("all", "export", "manifest"):
        log("== manifest")
        write_manifest()
    if a.step in ("all", "verify"):
        log("== verify")
        res = verify(runs)
        (WORK / "verify.json").write_text(json.dumps(res, indent=1, default=str))
    if a.step in ("all", "tables"):
        log("== reference tables")
        reference_tables()
    save_sha_cache()
    if a.step in ("all", "lock"):
        log("== lock")
        lock()
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
