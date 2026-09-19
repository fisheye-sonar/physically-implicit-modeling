"""dispatch_sim.py — the greedy list schedule the dispatcher and plan.py share (stdlib only).

``schedule(jobs, fixed, free_at)``: ``jobs`` are queue definitions; ``fixed`` maps a job id to
``(host, end_h)`` for jobs already running or done (hours from now; done = 0); ``free_at`` is when
each host's GPU lane is next free (hours from now). Queued jobs are placed in priority order
(ties: the longer job first) on the allowed host that finishes them earliest, after their deps
(and the host's own ``host_deps``) are done. CPU-lane jobs (transfers) run beside the GPU lane on
their host and never block it. Returns per-job (host, start, end, est) and per-host finish times.
"""
from __future__ import annotations


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
    pending = [j for j in jobs if j["id"] not in fixed]
    pending.sort(key=lambda j: (j.get("priority", 50), -max(j.get("est_hours", {"lab": 0}).values()), j["id"]))
    guard = 0
    while pending and guard < 10_000:
        guard += 1
        progress = False
        for j in list(pending):
            deps = j.get("deps", [])
            if any(d in by_id and d not in done_at for d in deps):
                continue
            best = None
            for host in j.get("hosts", ["lab"]):
                if host not in free_at:
                    continue
                hd = j.get("host_deps", {}).get(host, [])
                if any(d in by_id and d not in done_at for d in hd):
                    continue
                ready = max([lanes[(host, j.get("lane", "gpu"))]] + [done_at[d] for d in deps if d in done_at]
                            + [done_at[d] for d in hd if d in done_at])
                est = float(j.get("est_hours", {}).get(host, max(j.get("est_hours", {"lab": 1.0}).values())))
                end = ready + est
                if best is None or end < best[2]:
                    best = (host, ready, end, est)
            if best is None:
                continue
            host, start, end, est = best
            lanes[(host, j.get("lane", "gpu"))] = end
            done_at[j["id"]] = end
            placed[j["id"]] = {"host": host, "start": start, "end": end, "est": est, "fixed": False}
            pending.remove(j)
            progress = True
            break            # re-scan from the top so priority order is respected as lanes free up
        if not progress:
            for j in pending:   # unsatisfiable (a dep outside the queue, or no allowed host)
                placed[j["id"]] = {"host": None, "start": None, "end": None, "est": None, "fixed": False, "blocked": True}
            break
    hosts_end = {h: max([t for (hh, lane), t in lanes.items() if hh == h] + [0.0]) for h in free_at}
    ends = [p["end"] for p in placed.values() if p.get("end") is not None]
    return {"jobs": placed, "hosts": hosts_end, "end": max(ends) if ends else 0.0}
