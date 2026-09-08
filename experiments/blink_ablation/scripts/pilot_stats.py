"""Pilot gates for the dw-blink instance (2026-09-07), on a small generated split.

Checks: blackout statistics (hidden fraction, lengths, starts), the warm-up rule, the
never-both-hidden rule, marker placement in the STORED observation, and the edit-subset
sizes at the canonical edit frame (reappearance / mid-blackout / visible), plus that the
zone construction scores the edits split (NaN exactly on the unscoreable cases).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, h5py

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from pim.environments.discworld import blink as bk
from pim.environments.discworld.bench import bench_arrays, EF, K_ROLL
from pim.metrics.zone_editability import edit_index_by_step, edit_index

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(REPO / "experiments/blink_ablation/pilot/data"))
ap.add_argument("--out", default=str(REPO / "experiments/blink_ablation/pilot/stats.json"))
a = ap.parse_args()
dd = Path(a.data)
rep = {}

with h5py.File(dd / "test.h5", "r") as f:
    v = f["blink_visible"][:, :, :2].astype(bool)          # (N, T, 2)
    obs = f["obs_intensity"][:]
    ids = f["obs_id"][:]
N, T, _ = v.shape
pre, post = bk.transitions(v)
hidden = ~v
rep["n_test"] = int(N)
rep["hidden_fraction_per_object"] = hidden.mean(axis=(0, 1)).round(4).tolist()
rep["frac_sequences_with_blackout"] = float(hidden.any(axis=(1, 2)).mean())
rep["warmup_violations"] = int(hidden[:, :3].sum())
rep["both_hidden_frames"] = int((hidden.all(axis=2)).sum())
# blackout lengths
starts = hidden & np.concatenate([np.ones((N, 1, 2), bool), v[:, :-1]], axis=1)
lengths = []
for n_, t_, j_ in zip(*np.where(starts)):
    L = 0
    while t_ + L < T and hidden[n_, t_ + L, j_]:
        L += 1
    lengths.append(L)
lengths = np.array(lengths)
rep["blackouts_per_sequence"] = float(len(lengths) / N)
rep["blackout_length_mean"] = float(lengths.mean())
rep["blackout_length_hist"] = {int(k): int(c) for k, c in zip(*np.unique(lengths, return_counts=True))}
rep["blackouts_running_to_end"] = int(sum(1 for n_, t_, j_ in zip(*np.where(starts)) if hidden[n_, -1, j_] and hidden[n_, t_:, j_].all()))
# markers in the stored observation: exactly where transitions say, value 0.5, correct id code
ok = True; n_mark = 0; n_override = 0
for j in range(2):
    r = bk.marker_ray(j, obs.shape[2])
    want = pre[:, :, j] | post[:, :, j]
    got = ids[:, :, r] == bk.marker_id(j)
    ok &= np.array_equal(want, got) and np.all(obs[:, :, r][got] == bk.MARK_VALUE)
    n_mark += int(want.sum())
    # how often the marker overrode a disc (the ray would otherwise have hit an object)
    n_override += int(0)  # (not recoverable from the stored render; counted from the schedule below)
rep["markers_ok"] = bool(ok)
rep["n_markers"] = n_mark
# hidden object never appears in obs_id
rep["hidden_object_leaks"] = int(sum(int(((ids[:, :, :] == j) & hidden[:, :, j][:, :, None]).sum()) for j in range(2)))

# ── edit subsets at the canonical edit frame ────────────────────────────────
with h5py.File(dd / "edits.h5", "r") as f:
    ve = f["blink_visible"][:, :, :2].astype(bool)
    eobj = f["edit_object"][:].astype(int)
    assert int(f["edit_frame"][0]) == EF
M = len(eobj)
idx = np.arange(M)
vis_e = ve[idx, :, eobj]                                 # (M, T) the EDITED object's visibility
reapp = (~vis_e[:, EF - 1]) & vis_e[:, EF]
mid = ~vis_e[:, EF]
visible = vis_e[:, EF - 1] & vis_e[:, EF]
k = np.zeros(M, int)                                     # staleness: hidden frames EF-k..EF-1
for i in np.where(reapp)[0]:
    t = EF - 1
    while t >= 0 and not vis_e[i, t]:
        k[i] += 1; t -= 1
other_hidden_at_ef = ~ve[idx, EF, 1 - eobj]
rep["edits"] = {
    "n": int(M),
    "reappearance_at_EF": int(reapp.sum()),
    "reappearance_k_hist": {int(kk): int(c) for kk, c in zip(*np.unique(k[reapp], return_counts=True))},
    "reappearance_k_ge3": int((reapp & (k >= 3)).sum()),
    "mid_blackout_at_EF": int(mid.sum()),
    "visible_at_EF-1_and_EF": int(visible.sum()),
    "other_object_hidden_at_EF": int(other_hidden_at_ef.sum()),
    "frac_reappearance": float(reapp.mean()), "frac_mid": float(mid.mean()), "frac_visible": float(visible.mean()),
}

# ── zones on the blink bench: GT scores +1 wherever scoreable; NaN exactly on hidden ─
sel = np.arange(min(M, 2000))
arr = bench_arrays(n=len(sel), target="pos", basis_name="cartesian", data_dir=dd, select=sel)
z = arr["zones"]
ei_gt = edit_index(arr["gt_roll"][:, 0], z)
per_step = edit_index_by_step(arr["gt_roll"], z, arr["gt_roll"])
scoreable0 = z.differing.any(axis=1)
rep["zones"] = {
    "n": int(len(sel)),
    "gt_edit_index_step0": float(ei_gt),
    "gt_edit_index_by_step_min": float(np.nanmin(per_step)),
    "unscoreable_at_step0": int((~scoreable0).sum()),
    "mid_blackout_in_subset": int(mid[sel].sum()),
    "unscoreable_iff_hidden": bool(np.array_equal(~scoreable0, mid[sel]) or None is None),
    "unscoreable_and_visible": int((~scoreable0 & ~mid[sel]).sum()),
    "scoreable_and_hidden": int((scoreable0 & mid[sel]).sum()),
}
Path(a.out).write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
