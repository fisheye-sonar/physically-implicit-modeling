"""dispatch_sim.py — the forecast the dispatcher and plan.py share (stdlib only).

``schedule(jobs, fixed, free_at)``: ``jobs`` are queue definitions; ``fixed`` maps a job id to
``(host, end_h)`` for jobs already running or done (hours from now; done = 0); ``free_at`` is when
each host's GPU lane is next free (hours from now). Returns per-job (host, start, end, est) and
per-host finish times.

2026-09-20: the forecast MIRRORS THE DISPATCHER'S LAUNCH RULE (``dispatch.py`` step 3) instead of
placing each job on the host that would finish it earliest: whichever lane frees first takes the
highest-priority job runnable at that moment — (priority, this host's position in the job's host
list, the longer job first) among the queued jobs that list the host and whose deps (and the host's
own ``host_deps``) are done; a lane with nothing runnable waits for the first dep to finish. The
earliest-finish rule let a job "wait" for the faster host, which the dispatcher never does, and the
listed ETA flipped by ~2.5 h whenever an estimate moved. CPU-lane jobs (transfers, the
categorical-IM catch-up) run beside the GPU lane on their host and never block it; a RUNNING
cpu-lane job occupies its lane until its forecast end.
"""
from __future__ import annotations


def _est(j: dict, host: str) -> float:
    eh = j.get("est_hours", {}) or {}
    return float(eh.get(host, max(eh.values()) if eh else 1.0))


def schedule(jobs: list[dict], fixed: dict, free_at: dict) -> dict:
    by_id = {j["id"]: j for j in jobs}
    done_at: dict[str, float] = {}
    placed: dict[str, dict] = {}
    lanes = {(h, "gpu"): float(t) for h, t in free_at.items()}
    for h in free_at:
        lanes.setdefault((h, "cpu"), 0.0)
    for jid, (host, end) in fixed.items():
        done_at[jid] = float(end)
        placed[jid] = {"host": host, "start": None, "end": float(end), "est": None, "fixed": True}
        lane = by_id.get(jid, {}).get("lane", "gpu")
        if end and (host, lane) in lanes:                 # a running job holds its lane until it ends
            lanes[(host, lane)] = max(lanes[(host, lane)], float(end))
    pending = {j["id"]: j for j in jobs if j["id"] not in fixed}
    guard = 0
    while pending and guard < 10_000:
        guard += 1
        best = None                                       # (start, pref, priority, -est, jid, host, lane)
        for (host, lane), t_free in lanes.items():
            cands = []
            for jid, j in pending.items():
                if j.get("lane", "gpu") != lane or host not in j.get("hosts", ["lab"]):
                    continue
                deps = [d for d in list(j.get("deps", [])) + list(j.get("host_deps", {}).get(host, [])) if d in by_id]
                if any(d not in done_at for d in deps):
                    continue                              # a dep not placed yet: not runnable on this pass
                ready = max([t_free] + [done_at[d] for d in deps])
                cands.append((ready, j.get("priority", 50), j["hosts"].index(host), -_est(j, host), jid))
            if not cands:
                continue
            t0 = min(c[0] for c in cands)                 # the moment this lane can launch anything at all
            _, prio, pref, negest, jid = sorted((c for c in cands if c[0] <= t0 + 1e-9),
                                                key=lambda c: (c[1], c[2], c[3], c[4]))[0]
            key = (t0, pref, prio, negest, jid, host, lane)
            if best is None or key < best:
                best = key
        if best is None:
            for jid in pending:   # unsatisfiable (a dep outside the queue, a failed dep, or no allowed host)
                placed[jid] = {"host": None, "start": None, "end": None, "est": None, "fixed": False, "blocked": True}
            break
        start, _, _, _, jid, host, lane = best
        est = _est(pending[jid], host)
        lanes[(host, lane)] = start + est
        done_at[jid] = start + est
        placed[jid] = {"host": host, "start": start, "end": start + est, "est": est, "fixed": False}
        del pending[jid]
    hosts_end = {h: max([t for (hh, lane), t in lanes.items() if hh == h] + [0.0]) for h in free_at}
    ends = [p["end"] for p in placed.values() if p.get("end") is not None]
    return {"jobs": placed, "hosts": hosts_end, "end": max(ends) if ends else 0.0}
