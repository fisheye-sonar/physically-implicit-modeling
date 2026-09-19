#!/usr/bin/env python3
"""dispatch.py — ONE TICK of the multi-machine job queue (experiments/paper_ci). A systemd user
timer on the lab box runs it every two minutes; it never depends on an interactive session.

Each tick: probe every host (GPU, disk, our ``pimci-*`` units, the run records and progress files
of the jobs running there) → fold finished jobs in (sync a remote job's outputs back, retry a
failed attempt up to ``max_attempts``, ping the alert channel with the numbers) → launch the
next runnable job on every free lane (rsync a remote job's inputs first; ``state/PAUSED`` holds
launches) → render ``dashboard/state.json`` (+ the ledger when something finished) → digest
every N hours on the digest channel, alerts on the alert channel (host unreachable, disk low,
launch failed, stall). The lab box is the source of truth for every run directory; a remote
job's outputs are rsynced back the moment it finishes and its inputs pushed right before it
starts. stdlib only.

    python experiments/paper_ci/scripts/dispatch.py            # one tick
    python experiments/paper_ci/scripts/dispatch.py --dry      # probe + render, launch nothing
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
Q = ROOT / "experiments" / "paper_ci"
STATE, QUEUE, DASH = Q / "state", Q / "queue", Q / "dashboard"
LOGDIR = ROOT / "logs" / "paper_ci"
sys.path.insert(0, str(Q / "scripts"))
from dispatch_sim import schedule  # noqa: E402

CFG = json.loads((Q / "config.json").read_text())
HOSTS = CFG["hosts"]
NOW = time.time()


# ── small utilities ──────────────────────────────────────────────────────────

def log(msg: str) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line)
    with open(LOGDIR / "dispatch.log", "a") as f:
        f.write(line + "\n")


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def event(kind: str, msg: str, **kw) -> None:
    with open(STATE / "events.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.time(), "iso": time.strftime("%FT%T"), "kind": kind, "msg": msg, **kw}) + "\n")
    log(f"{kind}: {msg}")


def ping(topic: str, title: str, body: str, tags: str = "information_source", priority: str = "default") -> None:
    url = CFG["ntfy"][topic]
    try:
        subprocess.run(["curl", "-sS", "--max-time", "20", "-H", f"Title: {title}", "-H", f"Tags: {tags}",
                        "-H", f"Priority: {priority}", "-d", body[:3500], url], capture_output=True, timeout=30)
    except Exception as e:  # pragma: no cover
        log(f"ping failed: {e}")


def once_per(key: str, hours: float) -> bool:
    """Rate limiter for repeated alerts: True at most once per ``hours`` for ``key``."""
    p = STATE / "ratelimit" / f"{key}.ts"
    last = float(p.read_text()) if p.exists() else 0.0
    if NOW - last < hours * 3600:
        return False
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(NOW))
    return True


def sh(cmd: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def host_cmd(host: str, cmd: str, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run a shell command in the host's repo (locally, or over ssh)."""
    H = HOSTS[host]
    full = f"cd {shlex.quote(H['repo'])} && {cmd}"
    if H["ssh"]:
        return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", H["ssh"], full], timeout)
    return sh(["bash", "-lc", full], timeout)


def rsync(src: str, dst: str, timeout: float = 3600) -> subprocess.CompletedProcess:
    return sh(["rsync", "-a", "--partial", src, dst], timeout)


def expand_lab(rel: str) -> list[str]:
    """Expand a glob relative to the lab repo (the source of truth)."""
    hits = sorted(glob.glob(str(ROOT / rel)))
    return [str(Path(h).relative_to(ROOT)) for h in hits]


def push_inputs(host: str, job: dict) -> bool:
    H = HOSTS[host]
    if not H["ssh"]:
        return True
    # the job's definition, the queue scripts and the config go first: the wrapper on the remote
    # reads the job file from ITS OWN tree (the 2026-09-18 smoke: "no such job")
    for rel in ("experiments/paper_ci/config.json", "experiments/paper_ci/scripts/", f"experiments/paper_ci/queue/{job['id']}.json"):
        cp = rsync(str(ROOT / rel), f"{H['ssh']}:{H['repo']}/{rel}", 300)
        if cp.returncode != 0:
            log(f"push {rel} -> {host} failed: {cp.stderr[-300:]}")
            return False
    for rel in job.get("inputs", []):
        for r in expand_lab(rel):
            src = str(ROOT / r) + ("/" if (ROOT / r).is_dir() else "")
            dst = f"{H['ssh']}:{H['repo']}/{r}" + ("/" if (ROOT / r).is_dir() else "")
            host_cmd(host, f"mkdir -p {shlex.quote(str(Path(r).parent))}", 60)
            cp = rsync(src, dst)
            if cp.returncode != 0:
                log(f"push {r} -> {host} failed: {cp.stderr[-300:]}")
                return False
    return True


def pull_outputs(host: str, job: dict) -> bool:
    H = HOSTS[host]
    ok = True
    if not H["ssh"]:
        return True
    rels = list(job.get("outputs", [])) + [f"logs/paper_ci/{job['id']}"]
    for rel in rels:
        if any(c in rel for c in "*?["):
            # expand the glob on the remote side
            cp = host_cmd(host, f"ls -d {rel} 2>/dev/null", 60)
            hits = [l.strip() for l in cp.stdout.splitlines() if l.strip()]
        else:
            hits = [rel]
        for r in hits:
            isdir = host_cmd(host, f"test -d {shlex.quote(r)} && echo d || echo f", 60).stdout.strip() == "d"
            src = f"{H['ssh']}:{H['repo']}/{r}" + ("/" if isdir else "")
            dst = str(ROOT / r) + ("/" if isdir else "")
            (ROOT / r).mkdir(parents=True, exist_ok=True) if isdir else (ROOT / r).parent.mkdir(parents=True, exist_ok=True)
            cp = rsync(src, dst)
            if cp.returncode != 0 and "No such file" not in cp.stderr:
                log(f"pull {r} <- {host} failed: {cp.stderr[-300:]}")
                ok = False
    return ok


# ── probing ──────────────────────────────────────────────────────────────────

def probe(host: str, ids: list[str]) -> dict | None:
    cp = host_cmd(host, f"PIM_CI_HOST={host} .pim/bin/python experiments/paper_ci/scripts/hostprobe.py {' '.join(ids)}", 90)
    if cp.returncode != 0 or not cp.stdout.strip():
        log(f"probe {host} failed rc={cp.returncode}: {cp.stderr[-200:]}")
        return None
    try:
        return json.loads(cp.stdout.strip().splitlines()[-1])
    except Exception as e:
        log(f"probe {host} unparsable: {e}: {cp.stdout[-200:]}")
        return None


def unit_name(jid: str) -> str:
    return f"pimci-{jid}"


# ── the tick ─────────────────────────────────────────────────────────────────

def load_jobs() -> dict:
    return {p.stem: read_json(p) for p in sorted(QUEUE.glob("*.json"))}


def load_states(jobs: dict) -> dict:
    return {jid: read_json(STATE / f"{jid}.json", {"status": "queued", "attempts": 0, "history": []}) for jid in jobs}


def save_state(jid: str, st: dict) -> None:
    atomic_json(STATE / f"{jid}.json", st)


def progress_of(job: dict, hp: dict | None, st: dict) -> dict:
    """Live progress for a running training job from its metrics tail; ETA in hours."""
    out = {}
    if not hp or not job.get("progress"):
        return out
    j = hp.get("jobs", {}).get(job["id"], {})
    tail = j.get("metrics_tail") or []
    if tail:
        m = tail[-1]
        step, el = int(m.get("step", 0)), float(m.get("elapsed_s", 0) or 0)
        total = int(job["progress"]["steps"])
        rate = step / el if el > 0 else None
        # the first steps of a RESUMED run reuse the parent's elapsed clock, so the rate is honest
        rem_h = ((total - step) / rate / 3600) if rate else None
        out.update({"step": step, "total": total, "frac": step / total if total else None, "rate": rate,
                    "train_left_h": rem_h, "val_loss": m.get("val_loss"), "train_loss": m.get("train_loss")})
        if rem_h is not None:
            out["eta_h"] = rem_h + (job.get("n_scored", 1) * job.get("score_h", 0.5))
    out["metrics_mtime"] = j.get("metrics_mtime")
    out["log_mtime"] = j.get("log_mtime")
    return out


def tick(dry: bool = False) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    DASH.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs()
    states = load_states(jobs)
    paused = (STATE / "PAUSED").exists()
    hostinfo, unreachable = {}, {}
    hs = read_json(STATE / "hosts.json", {})     # persistent per-host bookkeeping (unreachable counts)

    # 1. probe every host with the ids of the jobs it is running
    for host in HOSTS:
        ids = [jid for jid, st in states.items() if st.get("status") == "running" and st.get("host") == host]
        hp = probe(host, ids)
        hostinfo[host] = hp
        rec = hs.setdefault(host, {"unreachable_ticks": 0, "last_seen": None})
        if hp is None:
            rec["unreachable_ticks"] += 1
            if rec["unreachable_ticks"] == CFG["unreachable_ticks"]:
                event("host_unreachable", f"{host} unreachable for {rec['unreachable_ticks']} ticks")
                ping("alerts", f"PIM CI: {HOSTS[host]['label']} UNREACHABLE", f"no probe for {rec['unreachable_ticks']} ticks; "
                     f"its running jobs continue on their own if the box is up; nothing new is launched there", "rotating_light", "high")
        else:
            if rec["unreachable_ticks"] >= CFG["unreachable_ticks"]:
                event("host_back", f"{host} reachable again")
                ping("alerts", f"PIM CI: {HOSTS[host]['label']} back", "reachable again; dispatch resumes there", "white_check_mark")
            rec["unreachable_ticks"] = 0
            rec["last_seen"] = NOW
            rec["snapshot"] = {k: v for k, v in hp.items() if k != "jobs"}
            if hp.get("disk_free_gb") is not None and hp["disk_free_gb"] < CFG["disk_min_gb"] and once_per(f"disk_{host}", 6):
                ping("alerts", f"PIM CI: {HOSTS[host]['label']} disk low", f"{hp['disk_free_gb']} GB free (floor {CFG['disk_min_gb']})", "warning", "high")
            g = hp.get("gpu") or {}
            if g.get("error") and once_per(f"gpu_{host}", 1):
                ping("alerts", f"PIM CI: {HOSTS[host]['label']} GPU query failed", g["error"], "warning", "high")
    atomic_json(STATE / "hosts.json", hs)

    # 2. reconcile running jobs
    finished_now = []
    for jid, st in states.items():
        if st.get("status") != "running":
            continue
        job, host = jobs[jid], st.get("host")
        hp = hostinfo.get(host)
        if hp is None:
            st["note"] = f"host {host} unreachable at {time.strftime('%T')}"
            save_state(jid, st)
            continue
        run = hp.get("jobs", {}).get(jid, {}).get("run")
        unit = hp.get("units", {}).get(unit_name(jid))
        active = bool(unit and unit.get("active") in ("active", "activating", "deactivating"))
        st["progress"] = progress_of(job, hp, st)
        if active:
            # stall: an active unit whose progress file and log have not moved for stall_min
            mt = [t for t in (st["progress"].get("metrics_mtime"), st["progress"].get("log_mtime")) if t]
            ref = max(mt) if mt else st.get("started_ts", NOW)
            if NOW - ref > CFG["stall_min"] * 60 and once_per(f"stall_{jid}", 2):
                event("stall", f"{jid} on {host}: no progress for {int((NOW - ref) / 60)} min while the unit is active")
                ping("alerts", f"PIM CI: {jid} may be STALLED on {host}", f"unit active, no metrics/log change for {int((NOW - ref) / 60)} min. "
                     f"Check: systemctl --user status {unit_name(jid)} / logs/paper_ci/{jid}/", "warning", "high")
            save_state(jid, st)
            continue
        # not active: finished, failed, or died without a record
        rc = run.get("rc") if run else None
        ended = run.get("ended") if run else None
        st.setdefault("history", []).append({"attempt": st.get("attempts"), "host": host, "started": st.get("started"),
                                             "ended": ended or time.strftime("%FT%T"), "rc": rc,
                                             "duration_s": run.get("duration_s") if run else None,
                                             "unit": unit})
        if run and ended and rc == 0:
            synced = pull_outputs(host, job)
            st.update({"status": "done", "ended": ended, "ended_ts": run.get("ended_ts", NOW), "rc": 0,
                       "duration_s": run.get("duration_s"), "synced": synced})
            save_state(jid, st)
            finished_now.append(jid)
            event("done", f"{jid} on {host} in {(run.get('duration_s') or 0) / 3600:.2f} h" + ("" if synced else " (SYNC-BACK FAILED)"))
            continue
        why = f"rc {rc}" if (run and ended) else "unit gone without an end record (killed, OOM-capped or rebooted)"
        st["attempts"] = int(st.get("attempts", 0))
        if st["attempts"] < job.get("max_attempts", 3):
            st.update({"status": "queued", "retry_after": NOW + 120, "last_error": why})
            event("retry", f"{jid} on {host} failed ({why}); attempt {st['attempts']}/{job.get('max_attempts', 3)} — requeued")
            if not (run and ended):      # the wrapper pings its own failures; this one had no wrapper record
                ping("alerts", f"PIM CI: {jid} died on {host}", f"{why}; requeued (attempt {st['attempts']})", "warning", "high")
        else:
            st.update({"status": "failed", "last_error": why, "ended": ended or time.strftime("%FT%T")})
            event("failed", f"{jid} FAILED permanently after {st['attempts']} attempts ({why})")
            ping("alerts", f"PIM CI: {jid} FAILED PERMANENTLY", f"{st['attempts']} attempts on {host}; last: {why}. "
                 f"Fix and reset: rm experiments/paper_ci/state/{jid}.json", "rotating_light", "urgent")
        save_state(jid, st)

    # 3. launch on every free lane
    launched = []
    if not dry and not paused:
        for host, H in HOSTS.items():
            hp = hostinfo.get(host)
            if hp is None:
                continue
            for lane in ("gpu", "cpu"):
                busy = [jid for jid, st in states.items() if st.get("status") == "running" and st.get("host") == host
                        and jobs[jid].get("lane", "gpu") == lane]
                if busy:
                    continue
                cands = []
                for jid, st in states.items():
                    job = jobs[jid]
                    if st.get("status") != "queued" or job.get("lane", "gpu") != lane or host not in job.get("hosts", []):
                        continue
                    if job.get("hold") or st.get("hold") or st.get("retry_after", 0) > NOW:
                        continue
                    deps = list(job.get("deps", [])) + list(job.get("host_deps", {}).get(host, []))
                    if any(states.get(d, {}).get("status") != "done" for d in deps if d in jobs):
                        continue
                    # a job both hosts could take goes to the host it lists first, unless that host is
                    # busy/unreachable — the other host takes it only if nothing else is queued for it
                    pref = job["hosts"].index(host)
                    cands.append((job.get("priority", 50), pref, -max(job.get("est_hours", {"lab": 0}).values()), jid))
                if not cands:
                    continue
                cands.sort()
                # if this host is the second choice for the top candidate and its first choice is free
                # and reachable, leave it to the first choice unless a first-choice job exists here
                prio, pref, _, jid = cands[0]
                if pref > 0:
                    first = jobs[jid]["hosts"][0]
                    first_free = hostinfo.get(first) is not None and not [k for k, s in states.items()
                                                                       if s.get("status") == "running" and s.get("host") == first
                                                                       and jobs[k].get("lane", "gpu") == lane]
                    own = [c for c in cands if c[1] == 0]
                    if first_free and own:
                        prio, pref, _, jid = own[0]
                job = jobs[jid]
                st = states[jid]
                st["attempts"] = int(st.get("attempts", 0)) + 1
                if not push_inputs(host, job):
                    event("launch_failed", f"{jid}: input sync to {host} failed")
                    st["attempts"] -= 1
                    st["retry_after"] = NOW + 600
                    save_state(jid, st)
                    continue
                (LOGDIR / jid).mkdir(parents=True, exist_ok=True)
                mem = job.get("mem_max") or H["mem_max"]
                unit = unit_name(jid)
                launch = (f"mkdir -p logs/paper_ci/{jid} && systemctl --user reset-failed {unit}.service 2>/dev/null; "
                          f"systemd-run --user --unit={unit} -p MemoryMax={mem} --collect "
                          f"--setenv=PIM_CI_HOST={host} --setenv=PIM_CI_UNIT={unit} --working-directory={shlex.quote(H['repo'])} "
                          f"/usr/bin/bash -c {shlex.quote(f'bash experiments/paper_ci/scripts/run_job.sh {jid} {st['attempts']} > logs/paper_ci/{jid}/unit.log 2>&1')}")
                cp = host_cmd(host, launch, 120)
                if cp.returncode != 0:
                    event("launch_failed", f"{jid} on {host}: {cp.stderr[-300:]}")
                    ping("alerts", f"PIM CI: launch of {jid} on {host} FAILED", cp.stderr[-500:] or cp.stdout[-500:], "rotating_light", "high")
                    st["attempts"] -= 1
                    st["retry_after"] = NOW + 600
                    save_state(jid, st)
                    continue
                st.update({"status": "running", "host": host, "unit": unit, "started": time.strftime("%FT%T"),
                           "started_ts": NOW, "progress": {}})
                st.pop("retry_after", None)
                save_state(jid, st)
                launched.append(jid)
                event("launch", f"{jid} on {host} (attempt {st['attempts']}, lane {lane}): {job['cmd']}")

    # 4. success pings, with the numbers, after the ledger is refreshed
    ledger = None
    if finished_now or not (DASH / "ledger.json").exists() or NOW - (DASH / "ledger.json").stat().st_mtime > 1800:
        ledger = refresh_ledger()
    for jid in finished_now:
        job = jobs[jid]
        st = states[jid]
        body = f"{job.get('note', '')}\n{(st.get('duration_s') or 0) / 3600:.2f} h on {st.get('host')}"
        if ledger and job.get("group") in ledger.get("groups", {}):
            body += "\n" + ledger_lines(ledger["groups"][job["group"]])
        rem = [k for k, s in states.items() if s.get("status") in ("queued", "running")]
        body += f"\n{len(rem)} jobs left · dashboard {CFG['dashboard_url']}"
        ping("alerts", f"PIM CI: {jid} DONE", body, "white_check_mark")

    # 5. render + digest
    render(jobs, states, hostinfo, hs, paused, ledger)
    digest_p = STATE / "last_digest.ts"
    last = float(digest_p.read_text()) if digest_p.exists() else 0.0
    if NOW - last > CFG["digest_hours"] * 3600:
        digest_p.write_text(str(NOW))
        ping("digest", "PIM CI status", digest_text(jobs, states, hostinfo), "hourglass_flowing_sand", "low")
    if launched or finished_now:
        log(f"launched {launched} · finished {finished_now}")


# ── ledger + rendering ───────────────────────────────────────────────────────

def refresh_ledger() -> dict | None:
    try:
        cp = sh([str(ROOT / ".pim/bin/python"), str(Q / "scripts" / "ledger.py")], timeout=600)
        if cp.returncode != 0:
            log(f"ledger failed: {cp.stderr[-400:]}")
        return read_json(DASH / "ledger.json")
    except Exception as e:
        log(f"ledger error: {e}")
        return read_json(DASH / "ledger.json")


def ledger_lines(group: dict) -> str:
    """One line per block for the completion ping: each editor's index AND guard as
    canonical (mean ± SD, n), plus the MLP skill."""
    out = []

    def cell(v):
        if not v or v.get("canonical") is None:
            return "—"
        if v.get("n", 0) >= 2 and v.get("sd") is not None:
            return f"{v['canonical']:+.3f} ({v['mean']:+.3f}±{v['sd']:.3f}, n{v['n']})"
        return f"{v['canonical']:+.3f} (n1)"

    for blk, metrics in group.get("blocks", {}).items():
        parts = [f"{ed}: EI {cell(metrics.get(f'{ed} EI'))} / fid {cell(metrics.get(f'{ed} fid'))}"
                 for ed in ("PI", "GS", "IM") if metrics.get(f"{ed} EI")]
        if metrics.get("skill_MLP"):
            parts.append(f"MLP {cell(metrics['skill_MLP'])}")
        if parts:
            out.append(f"[{blk}] " + " · ".join(parts))
    return "\n".join(out)


def eta_block(jobs: dict, states: dict, hostinfo: dict) -> dict:
    fixed, free_at = {}, {}
    for host in HOSTS:
        free_at[host] = 0.0
    for jid, st in states.items():
        if st.get("status") == "done":
            fixed[jid] = (st.get("host"), 0.0)
        elif st.get("status") == "running":
            p = st.get("progress") or {}
            job = jobs[jid]
            if p.get("eta_h") is not None:
                left = p["eta_h"]
            else:
                est = job.get("est_hours", {}).get(st.get("host"), 1.0)
                elapsed = (NOW - st.get("started_ts", NOW)) / 3600
                left = max(0.1, est - elapsed)
            fixed[jid] = (st.get("host"), left)
            if job.get("lane", "gpu") == "gpu":
                free_at[st["host"]] = max(free_at.get(st["host"], 0.0), left)
    for host in HOSTS:                       # an unreachable host takes nothing new in the forecast
        if hostinfo.get(host) is None and host in free_at:
            free_at[host] = max(free_at[host], 1e6)
    S = schedule([j for j in jobs.values() if states[j["id"]].get("status") != "failed"], fixed, free_at)
    return {"jobs": S["jobs"], "hosts": {h: (t if t < 1e5 else None) for h, t in S["hosts"].items()},
            "end_h": S["end"] if S["end"] < 1e5 else None,
            "end_iso": time.strftime("%FT%T", time.localtime(NOW + S["end"] * 3600)) if S["end"] < 1e5 else None}


def render(jobs, states, hostinfo, hs, paused, ledger) -> None:
    eta = eta_block(jobs, states, hostinfo)
    rows = []
    for jid, job in sorted(jobs.items(), key=lambda kv: (kv[1].get("priority", 50), kv[0])):
        st = states[jid]
        e = eta["jobs"].get(jid, {})
        rows.append({"id": jid, "group": job.get("group"), "kind": job.get("kind"), "hosts": job.get("hosts"),
                     "lane": job.get("lane", "gpu"), "priority": job.get("priority"), "note": job.get("note"),
                     "deps": job.get("deps", []), "host_deps": job.get("host_deps", {}), "est_hours": job.get("est_hours"),
                     "cmd": job.get("cmd"), "status": st.get("status"), "host": st.get("host"), "attempts": st.get("attempts", 0),
                     "started": st.get("started"), "ended": st.get("ended"), "duration_s": st.get("duration_s"),
                     "progress": st.get("progress"), "last_error": st.get("last_error"), "history": st.get("history", []),
                     "pred_host": e.get("host"), "pred_start_h": e.get("start"), "pred_end_h": e.get("end"),
                     "synced": st.get("synced"), "hold": bool(job.get("hold") or st.get("hold"))})
    hosts_out = {}
    for host, H in HOSTS.items():
        hp = hostinfo.get(host)
        rec = hs.get(host, {})
        snap = hp if hp is not None else rec.get("snapshot")
        running = [jid for jid, st in states.items() if st.get("status") == "running" and st.get("host") == host]
        hosts_out[host] = {"label": H["label"], "reachable": hp is not None, "last_seen": rec.get("last_seen"),
                           "unreachable_ticks": rec.get("unreachable_ticks", 0), "snapshot": {k: v for k, v in (snap or {}).items() if k != "jobs"},
                           "running": running, "eta_h": eta["hosts"].get(host),
                           "gpu_busy_h": sum((states[j].get("duration_s") or 0) for j in states if states[j].get("status") == "done"
                                             and states[j].get("host") == host and jobs[j].get("lane", "gpu") == "gpu") / 3600}
    events = []
    ev = STATE / "events.jsonl"
    if ev.exists():
        lines = ev.read_text().strip().splitlines()[-80:]
        events = [json.loads(l) for l in lines if l.strip()]
    counts = {}
    for st in states.values():
        counts[st.get("status", "queued")] = counts.get(st.get("status", "queued"), 0) + 1
    out = {"generated": time.time(), "generated_iso": time.strftime("%FT%T%z"), "paused": paused, "hosts": hosts_out,
           "jobs": rows, "counts": counts, "eta": {"end_h": eta["end_h"], "end_iso": eta["end_iso"], "hosts": eta["hosts"]},
           "ledger": ledger or read_json(DASH / "ledger.json"), "events": events[::-1],
           "config": {"digest_hours": CFG["digest_hours"], "stall_min": CFG["stall_min"], "dashboard_url": CFG["dashboard_url"]}}
    atomic_json(DASH / "state.json", out)


def digest_text(jobs, states, hostinfo) -> str:
    eta = eta_block(jobs, states, hostinfo)
    lines = []
    for host, H in HOSTS.items():
        hp = hostinfo.get(host)
        if hp is None:
            lines.append(f"{H['label']}: UNREACHABLE")
            continue
        g = hp.get("gpu") or {}
        run = [jid for jid, st in states.items() if st.get("status") == "running" and st.get("host") == host]
        prog = ""
        for jid in run:
            p = states[jid].get("progress") or {}
            if p.get("step"):
                prog += f" {jid} {p['step'] // 1000}k/{p['total'] // 1000}k (~{p.get('eta_h', 0):.1f} h left)"
            else:
                prog += f" {jid}"
        lines.append(f"{H['label']}: gpu {g.get('util', '?')}% {g.get('power_w', '?')} W · disk {hp.get('disk_free_gb', '?')} GB ·"
                     f"{prog or ' idle'}")
    c = {}
    for st in states.values():
        c[st.get("status", "queued")] = c.get(st.get("status", "queued"), 0) + 1
    lines.append(f"jobs: {c.get('done', 0)} done · {c.get('running', 0)} running · {c.get('queued', 0)} queued · {c.get('failed', 0)} failed")
    lines.append(f"ETA all done: {eta['end_iso'] or '?'} ({eta['end_h'] or 0:.1f} h)")
    lines.append(CFG["dashboard_url"])
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    lock = open(STATE / ".dispatch.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another tick is running; skipping")
        sys.exit(0)
    try:
        tick(dry=a.dry)
    except Exception as e:
        import traceback
        log("TICK CRASHED: " + traceback.format_exc()[-1500:])
        if once_per("tick_crash", 1):
            ping("alerts", "PIM CI: dispatcher tick crashed", f"{e!r}\nsee logs/paper_ci/dispatch.log", "rotating_light", "high")
        raise
