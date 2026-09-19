#!/usr/bin/env python3
"""plan.py — THE job list for the paper's seed-replicate queue (2026-09-18), written as
experiments/paper_ci/queue/<id>.json for the dispatcher.

    python experiments/paper_ci/plan.py            # write / refresh the queue (started jobs untouched)
    python experiments/paper_ci/plan.py --show     # print the plan and a greedy two-host schedule

What it encodes (Sevan, 2026-09-18): n = 3 training seeds per shortlist run — the parent's own
checkpoint at the matched budget (seed 0) plus two re-trainings — ONE budget everywhere, 512k
steps (L-oth-20m's saved checkpoints are 256k / 512k / 780k; every run has a 512k or 492,188
checkpoint). The sets that already exist at 390k (dw-noiseless, oth-adjacent-flip) are EXTENDED
to 512k by exact resume. Replicates are scored by the canonical pipeline in BOTH regression bases
and on the paper's categorical target (`appearance-fac`; the 8-ray grid sweep goes to the appendix
without a spread); the tables pool them into the parent row's ± at a matched budget. Data locality fixes the hosts: dw-128ray and dw-blink read
406 GB corpora that live only on the lab box; dw-16ray's corpus lives only on the remote; dw-8ray /
dw-5ray can go either way once their corpora are pushed (queue jobs, lab cpu lane); every Othello
corpus is on both. Priorities: the ray family first (the small contrasts), then the two Othello rows
quoted as levels, then blink, then the two inert Othello rows, then the cheap probe-seed extras,
then one tables rebuild on the lab.

Budgets are constants below; changing one and re-running this script rewrites only the jobs that
have not started.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
Q = REPO / "experiments" / "paper_ci"

STEPS_OTH = 512_000
STEPS_DW = 512_000          # Sevan 2026-09-18 evening: ONE budget everywhere (was 390k for discworld)
ENV_SCORE = {"PIM_DW_BASES": "frustum,cartesian", "PIM_SKIP_TOPICS": "training_curve", "PIM_SCORE_ONLY": "1"}
REMOTE_REPO = "research/physically-implicit-modeling"   # relative to the remote's $HOME (rsync target)

# family -> parent run, budget, measured train hours at 5090 rates for that budget (metrics.jsonl
# rates: Othello 11.2 steps/s, 128-input discworld 27, 8/5-ray 27.5, 16-ray 19.5), scoring hours per
# member (Othello master_eval 33 min; discworld ~45 min incl. the categorical fit), the categorical
# targets replicate.sh fits, allowed hosts (preference order), and host-specific deps.
FAMILIES = {
    # discworld train hours at 512k on the LAB 5090 UNDER ITS 450 W CAP (measured 2026-09-18 21:50:
    # 23.3 steps/s on 5-ray, 15% below the uncapped 27.5): 128-input 6.2, 8/5-ray 6.1, 16-ray 8.6.
    # The 4090 runs discworld at the uncapped 5090's rate (19.9 steps/s on 16-ray: the loader bounds
    # it), so its discworld factor is 0.85/1.0 relative to the capped lab; Othello stays 0.77.
    "dw-8ray":     dict(parent="ray_ablation/L-dw-8ray-20m", steps=STEPS_DW, train_h=6.1, score_h=0.75, remote_rate=1.18,
                        targets=["appearance-fac"], hosts=["remote", "lab"],
                        host_deps={"remote": ["xfer_dw-8ray_train"]}, prio=20),
    "dw-5ray":     dict(parent="ray_ablation/L-dw-5ray-20m", steps=STEPS_DW, train_h=6.1, score_h=0.75, remote_rate=1.18,
                        targets=["appearance-fac"], hosts=["lab", "remote"],
                        host_deps={"remote": ["xfer_dw-5ray_train"]}, prio=21),
    "dw-16ray":    dict(parent="ray_ablation/L-dw-16ray-20m", steps=STEPS_DW, train_h=8.6, score_h=0.8, remote_rate=1.18,
                        targets=["appearance-fac"], hosts=["remote"], prio=22),
    "dw-128ray":   dict(parent="ray_ablation/L-dw-128ray-20m", steps=STEPS_DW, train_h=6.2, score_h=0.8,
                        targets=["appearance-fac"], hosts=["lab"], prio=23),
    "dw-noiseless": dict(parent="noise_ablation/L-dw-noiseless-20m", steps=STEPS_DW, score_h=0.8,
                         targets=["appearance-fac"], hosts=["lab"], prio=24,
                         # EXTENSIONS of the existing 390k members (1.25 h each); the parent's 512k
                         # checkpoint becomes the seed-0 member, the 421,875 one is left outside the pool
                         seed_train_h={1: 1.5, 2: 1.5}),
    "oth-standard": dict(parent="initial_othello_comparison/L-oth-20m", steps=STEPS_OTH, train_h=12.7, score_h=0.6,
                         targets=[], hosts=["lab", "remote"], prio=30),
    "oth-adjflip": dict(parent="adjacent_flip_ablation/L-oth-adjacent-flip-20m", steps=STEPS_OTH, score_h=0.6,
                        targets=[], hosts=["remote", "lab"], prio=31,
                        # EXTENSIONS of the existing 390k members: seed 2 sits at 160k, seed 1 at 390k
                        seed_train_h={2: 8.7, 1: 3.0}),
    "dw-blink":    dict(parent="blink_ablation/L-dw-blink-20m", steps=STEPS_DW, train_h=6.2, score_h=0.8,
                        targets=["appearance-fac"], hosts=["lab"], prio=40),
    "oth-adjacent": dict(parent="adjacency_ablation/L-oth-adjacent-20m", steps=STEPS_OTH, train_h=12.7, score_h=0.6,
                         targets=[], hosts=["remote", "lab"], prio=50),
    "oth-noflip":  dict(parent="flip_ablation/L-oth-noflip-20m", steps=STEPS_OTH, train_h=12.7, score_h=0.6,
                        targets=[], hosts=["lab", "remote"], prio=51),
}
# the two corpora worth pushing to the remote (49 / 40 GB, ~8 min each at 100 MB/s) so the 4090 can
# take 8-ray / 5-ray replicates; nothing else moves (dw-128ray / dw-blink corpora are 406 GB)
TRANSFERS = {"dw-8ray": 5, "dw-5ray": 6}


def transfer_job(inst: str, prio: int) -> dict:
    src = f"datasets/discworld/{inst}/train/"
    return {"id": f"xfer_{inst}_train", "group": inst, "kind": "transfer", "hosts": ["lab"], "lane": "cpu",
            "cmd": f"rsync -a --partial --info=stats2 {src} wsl-sevan:{REMOTE_REPO}/{src}",
            "env": {}, "deps": [], "host_deps": {}, "priority": prio,
            "est_hours": {"lab": 0.2}, "inputs": [], "outputs": [], "progress": None,
            "mem_max": "4G", "max_attempts": 3,
            "note": f"push the {inst} training corpus to the remote (the 4090 can then train its replicates)"}


def replicate_job(fam: str, F: dict, seed: int) -> dict:
    topic, pname = F["parent"].split("/")
    run = f"{pname}__seed{seed}"
    steps = F["steps"]
    train_h = F.get("seed_train_h", {}).get(seed, F.get("train_h"))
    # a family's two re-trainings are INDEPENDENT jobs (they may run on both hosts at once); each
    # lays out the seed-0 member if it is missing and scores whatever is unscored, so both may
    # score the member — a duplicated half hour, deterministic content, and the sync-back is
    # idempotent. Estimate 2 scored runs per job.
    n_scored = 2
    est_lab = train_h + n_scored * F["score_h"]
    targets = " ".join(F["targets"])
    kind = "extend" if "seed_train_h" in F else "replicate"
    return {
        "id": f"rep_{fam}_s{seed}", "group": fam, "kind": kind, "hosts": F["hosts"], "lane": "gpu",
        "cmd": f"bash scripts/drivers/replicate.sh {F['parent']} {seed} {steps} {targets}".rstrip(),
        "env": dict(ENV_SCORE), "deps": [], "host_deps": F.get("host_deps", {}),
        "priority": F["prio"],
        "est_hours": {"lab": round(est_lab, 2), "remote": round(train_h / F.get("remote_rate", 0.77) + n_scored * F["score_h"] / 0.8, 2)},
        "train_h_lab": train_h, "score_h": F["score_h"], "n_scored": n_scored,
        "inputs": [f"runs/{F['parent']}", f"runs/{topic}/{pname}__seed0_s*", f"runs/{topic}/{run}"],
        "outputs": [f"runs/{topic}/{run}", f"runs/{topic}/{pname}__seed0_s*", f"logs/rep_{pname}_s{seed}", "runs/MOVES.md"],
        "progress": {"metrics": f"runs/{topic}/{run}/metrics.jsonl", "steps": steps},
        "mem_max": None, "max_attempts": 3,
        "note": (f"{'extend' if kind == 'extend' else 'train'} {run} to {steps // 1000}k, "
                 f"lay out the seed-0 member if missing, score (both bases{', ' + targets if targets else ''})"),
    }


def probe_seed_jobs() -> list[dict]:
    """Cheap extras at the tail: 10 probe seeds on the two best-edited runs (the 'probe seed is
    negligible' claim replicated where the contrasts live) and on adjacent-flip's 512k members
    (its 390k probe seeds are parked with the old budget)."""
    jobs = [
        {"id": "pseed_oth-standard", "group": "oth-standard", "kind": "probe_seeds", "hosts": ["remote", "lab"], "lane": "gpu",
         "cmd": "python -u experiments/seed_variance/scripts/probe_seeds_othello.py --run initial_othello_comparison/L-oth-20m --seeds 10",
         "env": {}, "deps": [], "host_deps": {}, "priority": 60, "est_hours": {"lab": 0.6, "remote": 0.8},
         "inputs": ["runs/initial_othello_comparison/L-oth-20m"], "outputs": ["runs/initial_othello_comparison/L-oth-20m"],
         "progress": None, "mem_max": None, "max_attempts": 2, "note": "10 probe seeds (linear grid + inverse map) on standard Othello"},
        {"id": "pseed_dw-8ray", "group": "dw-8ray", "kind": "probe_seeds", "hosts": ["lab", "remote"], "lane": "gpu",
         "cmd": "python -u experiments/seed_variance/scripts/probe_seeds.py --run ray_ablation/L-dw-8ray-20m --targets full appearance-fac --seeds 10 6",
         "env": {}, "deps": [], "host_deps": {"remote": ["xfer_dw-8ray_train"]}, "priority": 61, "est_hours": {"lab": 1.5, "remote": 2.0},
         "inputs": ["runs/ray_ablation/L-dw-8ray-20m"], "outputs": ["runs/ray_ablation/L-dw-8ray-20m"],
         "progress": None, "mem_max": None, "max_attempts": 2, "note": "10 regression + 6 factorised probe seeds on dw-8ray"},
    ]
    P = "adjacent_flip_ablation/L-oth-adjacent-flip-20m"
    for k, member in ((1, f"{P}__seed1"), (2, f"{P}__seed2"), (0, f"{P}__seed0_s492188")):
        jobs.append({"id": f"pseed_oth-adjflip_s{k}", "group": "oth-adjflip", "kind": "probe_seeds", "hosts": ["remote", "lab"],
                     "lane": "gpu", "cmd": f"python -u experiments/seed_variance/scripts/probe_seeds_othello.py --run {member} --seeds 10",
                     "env": {}, "deps": ["rep_oth-adjflip_s1", "rep_oth-adjflip_s2"], "host_deps": {}, "priority": 62,
                     "est_hours": {"lab": 0.6, "remote": 0.8}, "inputs": [f"runs/{member}"], "outputs": [f"runs/{member}"],
                     "progress": None, "mem_max": None, "max_attempts": 2, "note": f"10 probe seeds on the 512k member {member.split('/')[-1]}"})
    return jobs


def build() -> list[dict]:
    jobs = [transfer_job(i, p) for i, p in TRANSFERS.items()]
    for fam, F in FAMILIES.items():
        for seed in (1, 2):
            jobs.append(replicate_job(fam, F, seed))
    jobs += probe_seed_jobs()
    rep_ids = [j["id"] for j in jobs if j["kind"] in ("replicate", "extend")]
    jobs.append({"id": "final_tables", "group": "tables", "kind": "score", "hosts": ["lab"], "lane": "gpu",
                 "cmd": "bash scripts/drivers/score_pending.sh paper_ci_final",
                 "env": {"PIM_DW_BASES": "frustum,cartesian", "PIM_SKIP_TOPICS": "training_curve"},
                 "deps": rep_ids, "host_deps": {}, "priority": 90, "est_hours": {"lab": 0.5},
                 "inputs": [], "outputs": [], "progress": None, "mem_max": None, "max_attempts": 2,
                 "note": "master_eval catches anything unscored, both table notebooks re-render with every ± in place"})
    return jobs


def write(jobs: list[dict]) -> None:
    (Q / "queue").mkdir(parents=True, exist_ok=True)
    kept, written = [], []
    for j in jobs:
        p = Q / "queue" / f"{j['id']}.json"
        st = Q / "state" / f"{j['id']}.json"
        if st.exists() and json.loads(st.read_text()).get("status") not in (None, "queued"):
            kept.append(j["id"])          # started, finished or failed: the definition is frozen
            continue
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(j, f, indent=1)
        os.replace(tmp, p)
        written.append(j["id"])
    print(f"wrote {len(written)} job files" + (f"; left {len(kept)} started jobs untouched: {kept}" if kept else ""))


def simulate(jobs: list[dict], rates: dict | None = None) -> dict:
    """Greedy list schedule on the two hosts from a cold start: per job (host, start, end), per host
    the finish time, the global finish. The dispatcher runs the same routine on the live state."""
    from dispatch_sim import schedule   # noqa: F401  (shared with the dispatcher)
    return schedule(jobs, {}, {"lab": 0.0, "remote": 0.0})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print the plan and a greedy schedule; write nothing")
    a = ap.parse_args()
    jobs = build()
    if a.show:
        import sys
        sys.path.insert(0, str(Q / "scripts"))
        from dispatch_sim import schedule
        S = schedule(jobs, {}, {"lab": 0.0, "remote": 0.0})
        print(f"{'id':<26} {'host':<7} {'prio':>4} {'est h':>6} {'start h':>8} {'end h':>7}  deps")
        for j in sorted(jobs, key=lambda j: S["jobs"][j["id"]]["start"]):
            s = S["jobs"][j["id"]]
            print(f"{j['id']:<26} {s['host']:<7} {j['priority']:>4} {s['est']:>6.1f} {s['start']:>8.1f} {s['end']:>7.1f}  {','.join(j['deps'])}")
        print(f"\nfinish: lab {S['hosts']['lab']:.1f} h · remote {S['hosts']['remote']:.1f} h · ALL {S['end']:.1f} h "
              f"({S['end'] / 24:.2f} days)")
    else:
        write(jobs)
