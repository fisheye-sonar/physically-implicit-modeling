import json
from collections import defaultdict
from pathlib import Path

SCR = Path(__file__).resolve().parent
R = json.loads((SCR / "inventory_runs.json").read_text())
PAPER_CAT = {"appearance-fac", "appearance", "grid-16x8", "grid-6x5", "grid-10x3", "pos@appearance"}
MB = 1e6


def probe_tiers(r):
    t = defaultdict(int)
    for p in r["probes"]:
        b = p["bytes"]
        t["all"] += b
        if p["cat"] == "req":
            t["req"] += b
            d = p["detail"]
            tgt = d.split()[-1].split("/")[0]
            paper = True
            if d.startswith(("fwd full/frustum", "IM full/frustum")):
                paper = False
            if d == "grid mine_signed":
                paper = False
            if d.startswith(("fwd cat", "IM cat", "fwd snapped")) and tgt not in PAPER_CAT:
                paper = False
            if paper:
                t["paper"] += b
        elif p["cat"] == "index":
            t["index"] += b
        else:
            t["dead"] += b
    return t


def classify_entry(name):
    if name == "best_model.pt":
        return "model"
    if name in ("config.json",):
        return "config"
    if name == "scores.json":
        return "scores"
    if name == "vocab.npz":
        return "vocab"
    if name == "probes/":
        return "probes"
    if name in ("ckpt/", "latest.pt"):
        return "ckpt"
    if name == "scores_backup/" or (name.startswith("scores.") and name.endswith(".json")):
        return "scores_old"
    if name == "variance.json":
        return "variance"
    if name == "figures/":
        return "figures"
    if name in ("metrics.jsonl", "commit_sha"):
        return "small_meta"
    if name.endswith(".json"):
        return "other_json"
    return "other"


rows = []
for r in R:
    g = defaultdict(int)
    names = defaultdict(list)
    for n, b in r["entries"].items():
        c = classify_entry(n)
        g[c] += b
        names[c].append(n)
    pt = probe_tiers(r)
    total = sum(r["entries"].values())
    minimal = g["model"] + g["config"] + g["scores"] + g["vocab"] + pt["req"] + g["small_meta"]
    minimal_tables = g["config"] + g["scores"]
    rows.append({"run": r["run"], "rep": r["replicate"], "total": total, **{k: g[k] for k in (
        "model", "ckpt", "probes", "scores", "scores_old", "variance", "figures", "other_json", "small_meta", "vocab", "config", "other")},
        "probes_req": pt["req"], "probes_paper": pt["paper"], "probes_dead": pt["dead"], "minimal": minimal,
        "minimal_tables": minimal_tables, "other_json_names": names["other_json"], "other_names": names["other"],
        "scores_old_names": names["scores_old"], "ckpt_n": r.get("ckpt_n"), "keys": r["best_model_keys"]})

(SCR / "run_table.json").write_text(json.dumps(rows, indent=1))

hdr = f"{'run':<58}{'total':>9}{'model':>7}{'ckpt':>8}{'probes':>8}{'p.req':>7}{'p.paper':>8}{'p.dead':>8}{'scores':>7}{'s.old':>7}{'var':>6}{'figs':>6}{'oth':>6}{'MIN':>7}"
print(hdr)
tot = defaultdict(float)
for x in rows:
    print(f"{x['run']:<58}{x['total']/MB:9.0f}{x['model']/MB:7.0f}{x['ckpt']/MB:8.0f}{x['probes']/MB:8.1f}{x['probes_req']/MB:7.1f}"
          f"{x['probes_paper']/MB:8.1f}{x['probes_dead']/MB:8.1f}{x['scores']/MB:7.2f}{x['scores_old']/MB:7.2f}{x['variance']/MB:6.2f}"
          f"{x['figures']/MB:6.1f}{x['other_json']/MB:6.2f}{x['minimal']/MB:7.0f}")
    grp = ("rep" if x["rep"] else ("extra" if any(s in x["run"] for s in ("smooth", "obs5", "tok")) else "main"))
    for k in ("total", "model", "ckpt", "probes", "probes_req", "probes_paper", "probes_dead", "scores", "scores_old",
              "variance", "figures", "other_json", "minimal", "minimal_tables"):
        tot[(grp, k)] += x[k]
    tot[(grp, "n")] += 1
print()
for grp in ("main", "extra", "rep"):
    print(grp, int(tot[(grp, 'n')]), {k: round(tot[(grp, k)] / MB, 1) for k in (
        "total", "model", "ckpt", "probes", "probes_req", "probes_paper", "probes_dead", "scores", "scores_old", "variance",
        "figures", "other_json", "minimal", "minimal_tables")})
print()
for x in rows:
    if x["other_json_names"] or x["other_names"]:
        print(x["run"], x["other_json_names"], x["other_names"])
print()
print(set(tuple(x["keys"]) for x in rows))
print({x["run"]: x["ckpt_n"] for x in rows if x["ckpt_n"]})
