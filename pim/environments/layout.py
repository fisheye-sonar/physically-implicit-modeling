"""pim.environments.layout — THE one place a path under ``datasets/`` is built (layout v2).

Spec: ``research/specs/DATASET_LAYOUT_SPEC.md`` (2026-09-10). One shape for both environment
classes; the directory names the split's ROLE, the file name carries its size:

    datasets/<class>/<inst>/
      instance.json             hand-written summary — never read by code
      layout.json               machine-written marker: {"version": 2, ...}
      train/                    the training corpus (discworld: obs.f32 + corpus.json;
                                othello: train_<n>.npz)
      probe/                    probe FIT corpora (the hold-out is an internal 80/20 split
                                by sequence): discworld probe_120k.h5 / probe_250k.h5 (+ .json
                                manifests); othello probe_<n>.npz, probe_large_<n>.npz + label caches
      eval/                     held-out sequences never used to fit anything: discworld
                                test.h5 (+ test.json); othello test_<n>.npz
      edits/v1/                 the current edit bench: discworld edits.h5 (+ edits.json,
                                selection.json); othello cases_<n>.pkl (+ .json)
      edits/v2/                 reserved for the paired-counterfactual bench
      tokens/                   frames-as-tokens (discworld, unchanged)
      _unused/                  files no code reads — moved, never deleted

Every function here resolves to the **v2** location once the instance carries
``layout.json``; until then it returns the **v1** (pre-migration) location, so code built on
this module runs unchanged on either tree. The fallback exists for the migration window
only (``scripts/migrate_datasets.py``).

Probe cache keys are LOGICAL (``probe_key``): ``data="discworld/<inst>"``,
``split="probe_120k"`` — never a filesystem path, so a move can no longer orphan a cache.
``legacy_probe_key`` maps the paths older callers still pass onto the same logical key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
LAYOUT_VERSION = 2
CLASSES = ("discworld", "othello")
DEFAULT_INSTANCE = {"discworld": "dw-pn04", "othello": "oth-uniform"}
DW_PROBE_SIZES = ("120k", "250k")
# othello split name -> its role directory (v2); every split lived in corpus/ under v1
OTH_ROLE = {"train": "train", "test": "eval", "probe": "probe", "probe_large": "probe"}
EDITS_VERSIONS = ("v1", "v2")

__all__ = [
    "REPO", "DATASETS", "LAYOUT_VERSION", "CLASSES", "DEFAULT_INSTANCE", "DW_PROBE_SIZES",
    "OTH_ROLE", "instance_root", "layout_file", "read_layout", "is_migrated", "write_marker",
    "ensure_marker", "train_dir", "probe_dir", "probe_file", "probe_manifest", "eval_dir",
    "eval_file", "eval_manifest", "edits_dir", "edits_file", "edits_manifest",
    "edits_selection", "othello_split_dir", "othello_split_file", "othello_cases_file",
    "tokens_dir", "unused_dir", "probe_key", "parse_dataset_path", "legacy_probe_key",
    "legacy_edits_instance",
]


# ── instance roots and the marker ────────────────────────────────────────────


def _check(cls: str) -> str:
    if cls not in CLASSES:
        raise KeyError(f"unknown environment class {cls!r}; one of {CLASSES}")
    return cls


def instance_root(cls: str, inst: str) -> Path:
    return DATASETS / _check(cls) / inst


def layout_file(cls: str, inst: str) -> Path:
    return instance_root(cls, inst) / "layout.json"


def read_layout(cls: str, inst: str) -> dict | None:
    p = layout_file(cls, inst)
    return json.loads(p.read_text()) if p.exists() else None


def is_migrated(cls: str, inst: str) -> bool:
    """True once the instance is in layout v2 (its ``layout.json`` says so)."""
    d = read_layout(cls, inst)
    return bool(d) and int(d.get("version", 0)) >= LAYOUT_VERSION


def write_marker(cls: str, inst: str, **extra) -> Path:
    """Write ``layout.json`` (version 2) for an instance. Extra fields (e.g. the move list)
    are recorded verbatim; an existing marker's fields are kept unless overridden."""
    p = layout_file(cls, inst)
    prev = read_layout(cls, inst) or {}
    prev.update({"version": LAYOUT_VERSION, "class": cls, "instance": inst,
                 "migrated": prev.get("migrated") or time.strftime("%Y-%m-%d %H:%M"), **extra})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(prev, indent=1))
    return p


def _has_v1_files(cls: str, inst: str) -> bool:
    r = instance_root(cls, inst)
    if cls == "discworld":
        return any((r / q).exists() for q in ("probe/test.h5", "probe_250k/test.h5", "eval/edits.h5"))
    return (r / "corpus").exists() or (r / "edits" / "cases_1001.pkl").exists()


def ensure_marker(cls: str, inst: str) -> Path:
    """Producers call this before writing: a NEW instance is born in layout v2. Refuses to
    stamp an instance that still holds v1 files — that one needs the migration."""
    if is_migrated(cls, inst):
        return layout_file(cls, inst)
    if _has_v1_files(cls, inst):
        raise RuntimeError(f"{cls}/{inst} holds layout-v1 files and no layout.json — run "
                           f"scripts/migrate_datasets.py before producing into it")
    return write_marker(cls, inst)


# ── the splits ───────────────────────────────────────────────────────────────


def train_dir(cls: str, inst: str) -> Path:
    """The training corpus directory. Discworld: unchanged across versions. Othello: the
    role dir under v2, ``corpus/`` under v1."""
    if cls == "othello":
        return othello_split_dir(inst, "train")
    return instance_root(_check(cls), inst) / "train"


def probe_dir(cls: str, inst: str) -> Path:
    """The v2 probe directory (discworld and othello). Under v1 discworld had two
    (``probe/`` and ``probe_250k/``) — use ``probe_file`` for a specific corpus."""
    if cls == "othello":
        return othello_split_dir(inst, "probe")
    return instance_root(_check(cls), inst) / "probe"


def probe_file(cls: str, inst: str, size: str = "120k") -> Path:
    """Discworld probe FIT corpus of the given size (``"120k"`` canonical regression recipe,
    ``"250k"`` the categorical-target / large-floor recipe)."""
    if cls != "discworld":
        raise ValueError("probe_file(size=…) is the discworld form; othello uses othello_split_file")
    if size not in DW_PROBE_SIZES:
        raise KeyError(f"probe size must be one of {DW_PROBE_SIZES}, got {size!r}")
    r = instance_root(cls, inst)
    if is_migrated(cls, inst):
        return r / "probe" / f"probe_{size}.h5"
    return r / ("probe" if size == "120k" else "probe_250k") / "test.h5"


def probe_manifest(cls: str, inst: str, size: str = "120k") -> Path:
    """The manifest beside a discworld probe corpus (its ``sim`` config and split record)."""
    f = probe_file(cls, inst, size)
    return f.with_suffix(".json") if is_migrated(cls, inst) else f.parent / "dataset.json"


def eval_dir(cls: str, inst: str) -> Path:
    if cls == "othello":
        return othello_split_dir(inst, "test")
    return instance_root(_check(cls), inst) / "eval"


def eval_file(cls: str, inst: str) -> Path:
    """Discworld's held-out sequences (``eval/test.h5``, both versions). Othello: use
    ``othello_split_file(inst, "test", n)``."""
    if cls != "discworld":
        raise ValueError("eval_file is the discworld form; othello uses othello_split_file")
    return instance_root(cls, inst) / "eval" / "test.h5"


def eval_manifest(cls: str, inst: str) -> Path:
    r = instance_root(_check(cls), inst)
    return r / "eval" / ("test.json" if is_migrated(cls, inst) else "dataset.json")


def edits_dir(cls: str, inst: str, version: str = "v1") -> Path:
    """Where an edit bench's files live. v2: ``edits/<version>/``. v1: discworld's bench was
    the ``eval/`` suite's edits split; othello's cases sat directly under ``edits/``."""
    if version not in EDITS_VERSIONS:
        raise KeyError(f"edits version must be one of {EDITS_VERSIONS}, got {version!r}")
    r = instance_root(_check(cls), inst)
    if is_migrated(cls, inst) or version != "v1":
        return r / "edits" / version
    return r / ("eval" if cls == "discworld" else "edits")


def edits_file(cls: str, inst: str, version: str = "v1", n_cases: int = 1001) -> Path:
    d = edits_dir(cls, inst, version)
    return d / "edits.h5" if cls == "discworld" else d / f"cases_{n_cases}.pkl"


def edits_manifest(cls: str, inst: str, version: str = "v1", n_cases: int = 1001) -> Path:
    d = edits_dir(cls, inst, version)
    if cls == "othello":
        return d / f"cases_{n_cases}.json"
    return d / ("edits.json" if is_migrated(cls, inst) else "dataset.json")


def edits_selection(cls: str, inst: str, version: str = "v1") -> Path:
    """An instance's FILTERED case list for the bench (exists for dw-8ray only; callers
    test ``.exists()``)."""
    if is_migrated(cls, inst):
        return edits_dir(cls, inst, version) / "selection.json"
    return instance_root(_check(cls), inst) / "edits_selection.json"


def othello_split_dir(inst: str, name: str) -> Path:
    if name not in OTH_ROLE:
        raise KeyError(f"othello split must be one of {sorted(OTH_ROLE)}, got {name!r}")
    r = instance_root("othello", inst)
    return r / OTH_ROLE[name] if is_migrated("othello", inst) else r / "corpus"


def othello_split_file(inst: str, name: str, n: int) -> Path:
    return othello_split_dir(inst, name) / f"{name}_{n}.npz"


def othello_cases_file(inst: str, n_cases: int = 1001, version: str = "v1") -> Path:
    return edits_file("othello", inst, version, n_cases)


def tokens_dir(inst: str) -> Path:
    return instance_root("discworld", inst) / "tokens"


def unused_dir(cls: str, inst: str) -> Path:
    return instance_root(_check(cls), inst) / "_unused"


# ── probe cache keys ─────────────────────────────────────────────────────────


def probe_key(cls: str, inst: str, size: str = "120k") -> tuple[str, str]:
    """The LOGICAL ``(data, split)`` fields of a discworld probe cache key. Path-free by
    design: a dataset move must never orphan a fitted probe again."""
    if size not in DW_PROBE_SIZES:
        raise KeyError(f"probe size must be one of {DW_PROBE_SIZES}, got {size!r}")
    return f"{_check(cls)}/{inst}", f"probe_{size}"


def parse_dataset_path(p) -> dict | None:
    """``{"cls", "inst", "rel"}`` for any path (absolute or ``datasets/``-relative) that lies
    inside an instance directory; None otherwise."""
    parts = Path(p).parts
    try:
        i = len(parts) - 1 - parts[::-1].index("datasets")
    except ValueError:
        return None
    # the LAST "datasets" component must be ours: absolute paths must match this repo
    if Path(p).is_absolute() and Path(*parts[: i + 1]) != DATASETS:
        return None
    rest = parts[i + 1:]
    if len(rest) < 2 or rest[0] not in CLASSES:
        return None
    return {"cls": rest[0], "inst": rest[1], "rel": tuple(rest[2:])}


def legacy_probe_key(p) -> tuple[str, str, str] | None:
    """``(cls, inst, size)`` for a probe-corpus path in the form older callers pass —
    ``<inst>/probe`` (the 120k corpus), ``<inst>/probe_250k`` (the 250k one), or the v2 file
    itself — else None (a pilot or test corpus outside the instances)."""
    d = parse_dataset_path(p)
    if d is None or d["cls"] != "discworld":
        return None
    rel = d["rel"]
    if rel == ("probe",) or rel == ("probe", "test.h5"):
        return d["cls"], d["inst"], "120k"
    if rel == ("probe_250k",) or rel == ("probe_250k", "test.h5"):
        return d["cls"], d["inst"], "250k"
    if len(rel) == 2 and rel[0] == "probe" and rel[1].startswith("probe_"):
        size = rel[1].split("probe_", 1)[1].split(".")[0]
        return (d["cls"], d["inst"], size) if size in DW_PROBE_SIZES else None
    return None


def legacy_edits_instance(p) -> tuple[str, str] | None:
    """``(cls, inst)`` when ``p`` is an instance's v1 edit-bench directory (``<inst>/eval``)
    or its v2 one (``<inst>/edits/v1``); else None."""
    d = parse_dataset_path(p)
    if d is None:
        return None
    if d["rel"] in (("eval",), ("edits", "v1")):
        return d["cls"], d["inst"]
    return None
