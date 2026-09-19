#!/usr/bin/env python3
"""hostprobe.py — one JSON snapshot of THIS host for the dispatcher (run locally or over ssh).

    python experiments/paper_ci/scripts/hostprobe.py [job id ...]

Reports: GPU (name, utilisation, memory, power, its compute processes), disk free at the
repo, uptime and load, the state of every ``pimci-*`` systemd user unit, and for each named
job its run record (``state/<id>.run.json``) plus the tail of its progress file (the job's
``progress.metrics``, a training ``metrics.jsonl``) so the dispatcher can compute a live ETA
without a second round trip. stdlib only; never raises — a probe that cannot read something
reports ``null`` for it.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
Q = ROOT / "experiments" / "paper_ci"


def sh(cmd: list[str], timeout: float = 20) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def gpu() -> dict | None:
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    if not Path(exe).exists():
        return None
    out = sh([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
              "--format=csv,noheader,nounits"])
    if not out or not out.strip():
        return {"error": "nvidia-smi failed"}
    try:
        name, util, mu, mt, pd_, pl, temp = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        procs = sh([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]) or ""
        return {"name": name, "util": float(util), "mem_used_mb": float(mu), "mem_total_mb": float(mt),
                "power_w": float(pd_), "power_limit_w": float(pl), "temp_c": float(temp),
                "n_procs": len([l for l in procs.strip().splitlines() if l.strip()])}
    except Exception as e:  # pragma: no cover
        return {"error": f"parse: {e}: {out[:120]}"}


def units() -> dict:
    out = sh(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", "pimci-*"]) or ""
    res = {}
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        name = parts[0].removesuffix(".service")
        st = sh(["systemctl", "--user", "show", f"{name}.service",
                 "-p", "ActiveState,SubState,Result,ExecMainStatus,ExecMainStartTimestamp,MemoryCurrent"]) or ""
        d = dict(l.split("=", 1) for l in st.splitlines() if "=" in l)
        res[name] = {"active": d.get("ActiveState"), "sub": d.get("SubState"), "result": d.get("Result"),
                     "rc": d.get("ExecMainStatus"), "since": d.get("ExecMainStartTimestamp"),
                     "mem_bytes": d.get("MemoryCurrent")}
    return res


def tail_metrics(rel: str, n: int = 1) -> list[dict]:
    p = ROOT / rel
    if not p.exists():
        return []
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().decode(errors="ignore").strip().splitlines()
        out = []
        for l in lines[-n:]:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
        return out
    except Exception:
        return []


def _expand(pattern: str) -> list[str]:
    import glob
    hits = glob.glob(str(ROOT / pattern))
    return [str(Path(h).relative_to(ROOT)) for h in hits] if any(c in pattern for c in "*?[") else [pattern]


def newest_mtime(paths: list[Path], cap: int = 20000) -> float | None:
    """The newest file mtime under the given files/dirs (recursive, at most ``cap`` entries)."""
    best, n = None, 0
    for p in paths:
        if not p.exists():
            continue
        it = [p] if p.is_file() else p.rglob("*")
        for f in it:
            n += 1
            if n > cap:
                return best
            try:
                if f.is_file():
                    m = f.stat().st_mtime
                    if best is None or m > best:
                        best = m
            except OSError:
                pass
    return best


def main():
    ids = sys.argv[1:]
    d = {"host": os.environ.get("PIM_CI_HOST") or socket.gethostname(), "ts": time.time(),
         "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": gpu(), "units": units(),
         "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1)}
    try:
        d["uptime_s"] = float(open("/proc/uptime").read().split()[0])
        d["load1"] = os.getloadavg()[0]
    except Exception:
        d["uptime_s"], d["load1"] = None, None
    jobs = {}
    for jid in ids:
        j = {}
        rec = Q / "state" / f"{jid}.run.json"
        if rec.exists():
            try:
                j["run"] = json.loads(rec.read_text())
            except Exception:
                j["run"] = None
        qf = Q / "queue" / f"{jid}.json"
        if qf.exists():
            try:
                job = json.loads(qf.read_text())
                prog = job.get("progress") or {}
                if prog.get("metrics"):
                    j["metrics_tail"] = tail_metrics(prog["metrics"], 1)
                    mp = ROOT / prog["metrics"]
                    j["metrics_mtime"] = mp.stat().st_mtime if mp.exists() else None
                # "progress" = the newest write among the wrapper's logs AND everything the job
                # produces (its outputs: run dirs, the replicate driver's own log dir) — the scoring
                # stages write there, not to the training metrics (false stall alarm 2026-09-19 03:22)
                j["log_mtime"] = newest_mtime([ROOT / "logs" / "paper_ci" / jid] +
                                              [ROOT / o for pat in job.get("outputs", []) for o in _expand(pat)])
            except Exception:
                pass
        jobs[jid] = j
    d["jobs"] = jobs
    print(json.dumps(d))


if __name__ == "__main__":
    main()
