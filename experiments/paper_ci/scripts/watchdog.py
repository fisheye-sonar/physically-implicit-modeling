#!/usr/bin/env python3
"""watchdog.py — the INDEPENDENT safety net on every host (a systemd user timer, every 5 min).

It shares nothing with the dispatcher but the job files: if the dispatcher, the lab box or the
network is gone, this still speaks. Checks on THIS host: a ``pimci-*`` unit in the failed state;
an active unit whose job's progress file has not moved for ``stall_min``; disk below the floor;
the GPU not answering. On the lab box: the dispatcher log not ticking. On a remote host: the lab
box not answering ``tailscale ping`` for three checks in a row (and its recovery). Every alert is
rate-limited per key so a persistent condition pings once, not every five minutes. stdlib only.

    python experiments/paper_ci/scripts/watchdog.py --host lab|remote
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
Q = ROOT / "experiments" / "paper_ci"
CFG = json.loads((Q / "config.json").read_text())
RL = Q / "state" / "watchdog"
NOW = time.time()


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        return subprocess.CompletedProcess(cmd, 1, "", str(e))


def ping(title, body, tags="warning", priority="high"):
    sh(["curl", "-sS", "--max-time", "20", "-H", f"Title: {title}", "-H", f"Tags: {tags}", "-H", f"Priority: {priority}",
        "-d", body[:2000], CFG["ntfy"]["alerts"]])


def once_per(key, hours):
    p = RL / f"{key}.ts"
    last = float(p.read_text()) if p.exists() else 0.0
    if NOW - last < hours * 3600:
        return False
    RL.mkdir(parents=True, exist_ok=True)
    p.write_text(str(NOW))
    return True


def counter(key, reset=False) -> int:
    p = RL / f"{key}.n"
    n = 0 if reset or not p.exists() else int(p.read_text() or 0)
    if not reset:
        n += 1
    RL.mkdir(parents=True, exist_ok=True)
    p.write_text(str(n))
    return n


def main(host: str):
    label = CFG["hosts"].get(host, {}).get("label", host)
    # failed units
    out = sh(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", "--state=failed", "pimci-*"]).stdout
    for line in out.splitlines():
        unit = line.split()[0] if line.split() else ""
        if unit and once_per(f"failed_{unit}", 6):
            ping(f"PIM CI [{label}]: unit {unit} is in the FAILED state",
                 "the dispatcher retries it if attempts remain; check logs/paper_ci/<id>/", "rotating_light")
    # active units: stall on the job's progress file
    out = sh(["systemctl", "--user", "list-units", "--no-legend", "--plain", "--state=active", "pimci-*"]).stdout
    for line in out.splitlines():
        unit = line.split()[0] if line.split() else ""
        if not unit:
            continue
        jid = unit.removeprefix("pimci-").removesuffix(".service")
        job = Q / "queue" / f"{jid}.json"
        if not job.exists():
            continue
        j = json.loads(job.read_text())
        refs = []
        prog = (j.get("progress") or {}).get("metrics")
        if prog and (ROOT / prog).exists():
            refs.append((ROOT / prog).stat().st_mtime)
        lg = ROOT / "logs" / "paper_ci" / jid
        if lg.exists():
            refs += [p.stat().st_mtime for p in lg.glob("*.log")]
        rec = Q / "state" / f"{jid}.run.json"
        if rec.exists():
            try:
                refs.append(float(json.loads(rec.read_text()).get("started_ts") or 0))
            except Exception:
                pass
        if refs and NOW - max(refs) > CFG["stall_min"] * 60 and once_per(f"stall_{jid}", 2):
            ping(f"PIM CI [{label}]: {jid} looks STALLED", f"unit active, nothing written for {int((NOW - max(refs)) / 60)} min "
                 f"(progress file / logs). systemctl --user status {unit}")
    # disk
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < CFG["disk_min_gb"] and once_per("disk", 6):
        ping(f"PIM CI [{label}]: disk low", f"{free:.0f} GB free at {ROOT}")
    # gpu
    exe = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    cp = sh([exe, "--query-gpu=name", "--format=csv,noheader"])
    if cp.returncode != 0 and once_per("gpu", 1):
        ping(f"PIM CI [{label}]: nvidia-smi failing", (cp.stderr or cp.stdout)[-300:], "rotating_light")
    if host == "lab":
        dl = ROOT / "logs" / "paper_ci" / "dispatch.log"
        age = (NOW - dl.stat().st_mtime) / 60 if dl.exists() else 1e9
        if age > 10 and once_per("dispatcher", 0.5):
            ping("PIM CI [lab]: dispatcher NOT TICKING", f"dispatch.log last written {age:.0f} min ago; "
                 "systemctl --user status pimci-dispatch.timer pimci-dispatch.service", "rotating_light")
    else:
        cp = sh(["tailscale", "ping", "-c", "1", "--timeout", "5s", "sevan-ubuntu-lab"], 20)
        if cp.returncode != 0:
            n = counter("lab_unreachable")
            if n == 3 or (n > 3 and once_per("lab_unreachable_ping", 1)):
                ping(f"PIM CI [{label}]: LAB BOX UNREACHABLE for {n} checks", "the remote's running job continues; "
                     "no dispatch, no dashboard updates, no sync-back until the lab box is back", "rotating_light", "urgent")
        else:
            n = counter("lab_unreachable", reset=True)
            p = RL / "lab_unreachable_ping.ts"
            if p.exists():
                p.unlink()
                ping(f"PIM CI [{label}]: lab box reachable again", "tailscale ping ok", "white_check_mark", "default")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    main(ap.parse_args().host)
