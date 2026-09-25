"""Read-only identity-string scanner for the anonymous release audit.

Walks the in-scope code tree, parses notebooks (source vs outputs separately),
and prints file -> line -> category -> short snippet. Writes nothing in the repo.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/sevan/research/PIM/physically-implicit-modeling"
SCOPE = [
    "pim", "scripts", "tests",
    "notebooks/master_eval.ipynb",
    "notebooks/build_paper_tables_and_figs.ipynb",
    "notebooks/build_appendix_tables_and_figs.ipynb",
    "paper/figs/qualitative_edits", "paper/figs/qualitative_edits_othello",
    "paper/figs/qualitative_main", "paper/figs/paper_style.py",
    "pyproject.toml", "poetry.lock",
]
TEXT_EXT = {".py", ".sh", ".md", ".json", ".toml", ".lock", ".txt", ".ipynb", ".cfg", ".yaml", ".yml", ""}

CATS = [
    ("name", re.compile(r"sevan|brodjian|hobley|perona|pietro", re.I)),
    ("institution", re.compile(r"caltech|california institute", re.I)),
    ("geography", re.compile(r"pasadena|california|los angeles|\bSoCal\b|\b(?:PT|PDT|PST)\b")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z][\w.-]*")),
    ("owner@", re.compile(r"owner@", re.I)),
    ("abs_path", re.compile(r"(?<![\w.~])/(?:home|Users|tmp|mnt|media|scratch|data|root|srv|opt|net|nfs|work)/|[A-Za-z]:\\\\|~/")),
    ("hostname", re.compile(r"\bwsl\b|wsl[-_]|tailscale|\b4090\b|\b5090\b|\blab box\b|\bthe lab\b|\blab machine\b|hostname|\bssh\b|\brsync\b|\bremote\b", re.I)),
    ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("github", re.compile(r"github\.com|gitlab|github", re.I)),
    ("date", re.compile(r"\b20\d\d-\d\d-\d\d\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:, 20\d\d)?\b")),
    ("time", re.compile(r"\b\d{1,2}:\d\d(?: ?[ap]m)?\b", re.I)),
    ("attribution", re.compile(r"\(Sevan\)|Sevan'?s|Sevan (?:said|asked|picked|chose|wants)|\bthe user\b|\bmy\b|advisor|professor|\bprof\b|labmate|colleague|lab meeting", re.I)),
    ("first_person", re.compile(r"(?<![\w\[.])I(?:'m|'ve|'d|'ll| am| think| want| ran| will| was| picked| chose| asked)\b")),
    ("username", re.compile(r"\bsbrodjia\b|\buser(?:name)?\s*[:=]", re.I)),
]


def iter_files():
    for s in SCOPE:
        p = os.path.join(ROOT, s)
        if os.path.isfile(p):
            yield p
            continue
        for dp, dns, fns in os.walk(p):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for f in fns:
                yield os.path.join(dp, f)


def scan_text(label, text, out):
    for ln, line in enumerate(text.splitlines(), 1):
        for cat, rx in CATS:
            for m in rx.finditer(line):
                # 'I' false positives: roman numerals / variable names in code
                if cat == "attribution" and m.group(0) == "I" and not re.search(r"#|\"\"\"|'''|^\s*[A-Z]", line):
                    continue
                a = max(0, m.start() - 40)
                b = min(len(line), m.end() + 40)
                out.append((label, ln, cat, m.group(0), line[a:b].strip()))


def main():
    out = []
    binaries = []
    for f in iter_files():
        rel = os.path.relpath(f, ROOT)
        ext = os.path.splitext(f)[1]
        if ext == ".ipynb":
            nb = json.load(open(f))
            for ci, cell in enumerate(nb["cells"]):
                src = "".join(cell.get("source", []))
                scan_text(f"{rel}#cell{ci}:source", src, out)
                for oi, o in enumerate(cell.get("outputs", [])):
                    chunks = []
                    if "text" in o:
                        chunks.append("".join(o["text"]))
                    for k, v in o.get("data", {}).items():
                        if k.startswith("text/") or k == "application/json":
                            chunks.append("".join(v) if isinstance(v, list) else json.dumps(v))
                        elif k.startswith("image/"):
                            chunks.append(f"[{k} {len(v)} chars]")
                    if "traceback" in o:
                        chunks.append("\n".join(o["traceback"]))
                    scan_text(f"{rel}#cell{ci}:output{oi}", "\n".join(chunks), out)
            md = json.dumps(nb.get("metadata", {}))
            scan_text(f"{rel}#metadata", md, out)
        elif ext in TEXT_EXT:
            try:
                txt = open(f, encoding="utf-8").read()
            except UnicodeDecodeError:
                binaries.append(rel)
                continue
            scan_text(rel, txt, out)
        else:
            binaries.append(rel)
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode == "full":
        for r in out:
            print("\t".join([r[0], str(r[1]), r[2], r[3], r[4][:110]]))
    by_cat = Counter(r[2] for r in out)
    by_file_cat = defaultdict(Counter)
    for r in out:
        by_file_cat[r[0].split("#")[0]][r[2]] += 1
    print("\n=== counts per category ===", file=sys.stderr)
    for k, v in by_cat.most_common():
        print(f"{k}\t{v}", file=sys.stderr)
    print("\n=== per file ===", file=sys.stderr)
    for fn in sorted(by_file_cat):
        print(fn, dict(by_file_cat[fn]), file=sys.stderr)
    print("\n=== binary (not text-scanned) ===", file=sys.stderr)
    exts = Counter(os.path.splitext(b)[1] for b in binaries)
    print(dict(exts), file=sys.stderr)
    for b in binaries:
        if not b.endswith((".png", ".pdf")):
            print(b, file=sys.stderr)


main()
