import re
from collections import defaultdict
from pathlib import Path

SCR = Path(__file__).resolve().parent
files = defaultdict(dict)
cur = None
for line in (SCR / "datasets_files.txt").read_text().splitlines():
    m = re.match(r"=== (\S+)", line)
    if m:
        cur = m.group(1)
        continue
    sz, rel = line.split(None, 1)
    files[cur][rel.strip()] = int(sz)

MB = 1e6


def pick(inst, pred):
    return sum(s for f, s in files[inst].items() if pred(f))


rows = []
for inst, fs in files.items():
    oth = inst.startswith("othello")
    train = pick(inst, lambda f: f.startswith("train/"))
    unused = pick(inst, lambda f: f.startswith("_unused/"))
    meta = pick(inst, lambda f: f in ("instance.json", "layout.json"))
    if oth:
        tables = 0
        fig = pick(inst, lambda f: f in ("probe/probe_20000.npz", "edits/v1/cases_1000.pkl", "edits/v1/cases_1000.json"))
        rescore = pick(inst, lambda f: f in ("probe/probe_20000.npz", "eval/test_10000.npz", "edits/v1/cases_1000.pkl",
                                              "edits/v1/cases_1000.json"))
        baselines_refit = pick(inst, lambda f: f in ("probe/probe_large_170000.npz",))
        label_cache = pick(inst, lambda f: f == "probe/probe_large_170000_labels_170000.npz")
        dead = pick(inst, lambda f: f in ("edits/edits_10000.npz", "probe/probe_20000_labels_20000.npz",
                                           "probe/probe_large_170000_labels_40000.npz"))
        p120 = p250 = evalh5 = tok = 0
    else:
        tables = pick(inst, lambda f: f == "edits/v1/edits.h5")
        p120 = pick(inst, lambda f: f.startswith("probe/probe_120k"))
        p250 = pick(inst, lambda f: f.startswith("probe/probe_250k"))
        edits = pick(inst, lambda f: f.startswith("edits/v1/"))
        evalh5 = pick(inst, lambda f: f.startswith("eval/"))
        tokv = pick(inst, lambda f: f == "tokens/vocab.npz" or f == "tokens/meta.json")
        tok = pick(inst, lambda f: f == "tokens/train.i16")
        fig = tables + p120 + (p250 if inst.split("/")[1] in ("dw-16ray", "dw-8ray", "dw-5ray", "dw-128ray") else 0)
        rescore = edits + p120 + (p250 if inst.split("/")[1] in ("dw-128ray", "dw-16ray", "dw-8ray", "dw-5ray") else 0) + tokv
        baselines_refit = p250
        label_cache = 0
        dead = tok
    total = sum(fs.values())
    rows.append((inst, total, train, unused, dead, meta, tables, fig, rescore, baselines_refit, label_cache,
                 p120, p250, evalh5))
print(f"{'instance':<26}{'total':>9}{'train':>9}{'_unused':>9}{'dead':>8}{'tables':>8}{'figs':>8}{'rescore':>9}{'p120':>8}{'p250':>8}{'eval':>7}{'oth-lbl':>8}")
tot = defaultdict(float)
for r in rows:
    inst, total, train, unused, dead, meta, tables, fig, rescore, bref, lbl, p120, p250, ev = r
    print(f"{inst:<26}{total/MB:9.0f}{train/MB:9.0f}{unused/MB:9.1f}{dead/MB:8.1f}{tables/MB:8.1f}{fig/MB:8.1f}{rescore/MB:9.1f}{p120/MB:8.0f}{p250/MB:8.0f}{ev/MB:7.1f}{lbl/MB:8.0f}")
    grp = "oth" if inst.startswith("othello") else ("dw-extra" if inst.split("/")[1] in ("dw-smooth", "dw-8ray-obs5") else "dw-paper")
    for k, v in zip(("total", "train", "unused", "dead", "meta", "tables", "fig", "rescore", "bref", "lbl", "p120", "p250", "eval"), r[1:]):
        tot[(grp, k)] += v
for grp in ("oth", "dw-paper", "dw-extra"):
    print(grp, {k: round(tot[(grp, k)] / MB, 1) for k in ("total", "train", "unused", "dead", "meta", "tables", "fig", "rescore", "bref", "lbl", "p120", "p250", "eval")})
