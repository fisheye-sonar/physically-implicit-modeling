"""Read-only artifact metadata scanner (runs, baselines, datasets). Opens everything read-only.

Pickles (.pt zip archives, .pkl) are scanned at the opcode level: every GLOBAL (module, name)
and every string constant, without materialising tensors.
"""
import io
import json
import os
import pickletools
import re
import sys
import zipfile
from collections import Counter, defaultdict

ROOT = "/home/sevan/research/PIM/physically-implicit-modeling"
SCR = os.path.dirname(os.path.abspath(__file__))
RUNS = [l.strip() for l in open(os.path.join(SCR, "runs.txt")) if l.strip()]
DW = ["dw-noiseless", "dw-blink", "dw-128ray", "dw-16ray", "dw-8ray", "dw-5ray", "dw-smooth", "dw-8ray-obs5"]
OTH = ["oth-uniform", "oth-adjacent", "oth-adjacent-flip", "oth-noflip"]

ID = [
    ("abs_path", re.compile(r"(?<![\w.~])/(?:home|Users|tmp|mnt|media|scratch|data|root|srv|opt|net|nfs|work)/|[A-Za-z]:\\\\")),
    ("name", re.compile(r"sevan|brodjian|sbrodjia|hobley|perona", re.I)),
    ("institution/geo", re.compile(r"caltech|pasadena|california", re.I)),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z][\w.-]*")),
    ("host", re.compile(r"\bwsl|tailscale|owner@|\b(?:4090|5090)\b|hostname|\bhost\b|\bnode\b|\bmachine\b", re.I)),
    ("ip", re.compile(r"\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\b(?:192\.168|10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b")),
    ("date", re.compile(r"\b20\d\d-\d\d-\d\d(?:[T ]\d\d:\d\d(?::\d\d)?)?")),
]
RENAME = re.compile(r"discworld|Discworld|DISCWORLD|\bdw-|\bdw_|L-dw")

hits = defaultdict(Counter)          # category -> Counter(file-kind -> n)
examples = defaultdict(dict)         # category -> {snippet: first file}
rename = defaultdict(Counter)        # file-kind -> Counter(token)
json_keys_with_rename = defaultdict(Counter)   # file-kind -> Counter(json path)
globals_seen = defaultdict(Counter)  # file-kind -> Counter(module.name)


def kind_of(path):
    rel = os.path.relpath(path, ROOT)
    k = re.sub(r"probes_[0-9a-f]{16}", "probes_H", rel)
    k = re.sub(r"^runs/[^/]+/[^/]+/", "runs/*/*/", k)
    k = re.sub(r"^runs/_baselines/[^/]+/", "runs/_baselines/*/", k)
    k = re.sub(r"^datasets/(discworld|othello)/[^/]+/", r"datasets/\1/*/", k)
    k = re.sub(r"ckpt/[^/]+$", "ckpt/*", k)
    return k


def scan_str(s, path, where=""):
    kind = kind_of(path)
    for cat, rx in ID:
        for m in rx.finditer(s):
            hits[cat][kind] += 1
            a, b = max(0, m.start() - 30), min(len(s), m.end() + 50)
            snip = s[a:b].replace("\n", " ")
            if len(examples[cat]) < 40 and snip not in examples[cat]:
                examples[cat][snip] = os.path.relpath(path, ROOT) + where
    for m in RENAME.finditer(s):
        rename[kind][m.group(0)] += 1


def walk_json(o, path, jp="$"):
    kind = kind_of(path)
    if isinstance(o, dict):
        for k, v in o.items():
            if RENAME.search(str(k)):
                json_keys_with_rename[kind][f"KEY {re.sub(r'[0-9]+', 'N', jp)}.{k}"] += 1
            walk_json(v, path, f"{jp}.{k}" if not re.fullmatch(r"\d+|[0-9a-f]{8,}", str(k)) else f"{jp}.*")
    elif isinstance(o, list):
        for v in o[:50]:
            walk_json(v, path, jp + "[]")
    elif isinstance(o, str):
        if RENAME.search(o):
            json_keys_with_rename[kind][f"VAL {jp} = {o[:60]}"] += 1


def scan_text_file(path):
    try:
        s = open(path, encoding="utf-8").read()
    except Exception as e:
        print("unreadable", path, e)
        return
    scan_str(s, path)
    if path.endswith(".json"):
        try:
            walk_json(json.loads(s), path)
        except Exception:
            pass
    elif path.endswith(".jsonl"):
        for line in s.splitlines()[:3]:
            try:
                walk_json(json.loads(line), path)
            except Exception:
                pass


def pickle_strings(data, path):
    kind = kind_of(path)
    strs = []
    try:
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            if op.name in ("GLOBAL", "INST"):
                globals_seen[kind][arg.replace(" ", ".")] += 1
            elif op.name == "STACK_GLOBAL":
                pass
            elif isinstance(arg, str) and op.name.endswith("UNICODE") or op.name in ("STRING", "BINSTRING", "SHORT_BINSTRING"):
                strs.append(str(arg))
    except Exception as e:
        strs.append(f"<pickle parse error {e}>")
    # STACK_GLOBAL (protocol 4+) takes module/name from the two preceding strings
    try:
        prev = []
        for op, arg, pos in pickletools.genops(io.BytesIO(data)):
            if op.name == "STACK_GLOBAL" and len(prev) >= 2:
                globals_seen[kind][f"{prev[-2]}.{prev[-1]}"] += 1
            if isinstance(arg, str):
                prev.append(arg)
    except Exception:
        pass
    return strs


def scan_pickle_file(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            pk = [n for n in names if n.endswith("data.pkl")]
            for n in pk:
                strs = pickle_strings(z.read(n), path)
                scan_str("\n".join(strs), path, f"::{n}")
            # the archive's inner folder name is the file stem at save time
            scan_str("\n".join(names[:3]), path, "::zipnames")
            for n in names:
                if n.endswith((".json", "version", "byteorder")) or "/.data/" in n:
                    try:
                        scan_str(z.read(n)[:2000].decode("utf-8", "replace"), path, f"::{n}")
                    except Exception:
                        pass
    else:
        data = open(path, "rb").read()
        strs = pickle_strings(data, path)
        scan_str("\n".join(strs), path)


def scan_h5(path):
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            for k, v in obj.attrs.items():
                if isinstance(v, bytes):
                    v = v.decode("utf-8", "replace")
                if isinstance(v, str):
                    scan_str(v, path, f"::{name or '/'}@{k}")
                    out[f"{name or '/'}@{k}"] = v
                    if k.endswith("json"):
                        try:
                            walk_json(json.loads(v), path, f"$h5attr[{k}]")
                        except Exception:
                            pass
                else:
                    out[f"{name or '/'}@{k}"] = f"<{type(v).__name__}>"
        visit("", f)
        f.visititems(visit)
        out["__datasets__"] = [k for k in f.keys()]
    return out


def scan_npz(path):
    import numpy as np
    info = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            try:
                a = z[k]
            except ValueError as e:
                info[k] = f"<object array: {e}>"
                continue
            info[k] = f"{a.dtype}{a.shape}"
            if a.dtype.kind in "US" and a.size < 10000:
                s = "\n".join(str(x) for x in a.ravel()[:2000])
                scan_str(s, path, f"::{k}")
                if a.size == 1:
                    try:
                        walk_json(json.loads(str(a.ravel()[0])), path, f"$npz[{k}]")
                    except Exception:
                        pass
                info[k] += " = " + s[:200].replace("\n", " | ")
    return info


def main():
    h5_attrs, npz_info = {}, {}
    n_files = Counter()
    # ── runs ──
    for r in RUNS:
        base = os.path.join(ROOT, "runs", r)
        for dp, dns, fns in os.walk(base):
            for f in fns:
                p = os.path.join(dp, f)
                n_files[kind_of(p)] += 1
                if f.endswith((".json", ".jsonl", ".md")) or f == "commit_sha":
                    scan_text_file(p)
                elif f.endswith((".pt", ".pkl")):
                    if "/ckpt/" in p and not f.endswith("0.pt"):
                        pass
                    scan_pickle_file(p)
                elif f.endswith(".npz"):
                    npz_info[os.path.relpath(p, ROOT)] = scan_npz(p)
    # ── baselines ──
    for inst in DW + OTH:
        d = os.path.join(ROOT, "runs", "_baselines", inst)
        if not os.path.isdir(d):
            continue
        for dp, dns, fns in os.walk(d):
            for f in fns:
                p = os.path.join(dp, f)
                n_files[kind_of(p)] += 1
                if f.endswith((".json", ".jsonl", ".md")):
                    scan_text_file(p)
                elif f.endswith((".pt", ".pkl")):
                    scan_pickle_file(p)
    # ── datasets ──
    for cls, insts in (("discworld", DW), ("othello", OTH)):
        for inst in insts:
            d = os.path.join(ROOT, "datasets", cls, inst)
            for dp, dns, fns in os.walk(d):
                if "/_unused" in dp:
                    continue
                for f in fns:
                    p = os.path.join(dp, f)
                    rel = os.path.relpath(p, d)
                    if rel.startswith("train/") and not f.endswith((".json",)) and f != "meta.h5":
                        continue
                    n_files[kind_of(p)] += 1
                    if f.endswith((".json", ".md")):
                        scan_text_file(p)
                    elif f.endswith(".h5"):
                        h5_attrs[os.path.relpath(p, ROOT)] = scan_h5(p)
                    elif f.endswith(".npz"):
                        npz_info[os.path.relpath(p, ROOT)] = scan_npz(p)
                    elif f.endswith(".pkl"):
                        scan_pickle_file(p)
    res = {
        "n_files": dict(n_files),
        "hits": {c: dict(v) for c, v in hits.items()},
        "examples": examples,
        "rename_tokens": {k: dict(v) for k, v in rename.items()},
        "json_rename_paths": {k: dict(v.most_common(80)) for k, v in json_keys_with_rename.items()},
        "pickle_globals": {k: dict(v) for k, v in globals_seen.items()},
        "h5_attrs": h5_attrs,
        "npz_info": npz_info,
    }
    json.dump(res, open(os.path.join(SCR, "artifact_scan.json"), "w"), indent=1, default=str)
    print("done")


main()
