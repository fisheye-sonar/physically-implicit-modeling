"""Build the FILTERED edit-case selection for the frames-as-tokens dw-8ray bench.

The generated edits split is kept as it is (same generator, same seeds, same provenance); this
picks the subset of cases that are actually scoreable as a frame edit — the edited and
unedited worlds must render DIFFERENT frames, both in the vocabulary, and differ on at least
`--min-rays` rays. 17% of cases change no ray at all (mean teleport 0.92 vs 2.42 world units)
and another 22% change a single ray of eight. Writes the indices + provenance next to the
instance; `bench.bench_arrays` reads it — the ray-zone AND token benches, so the interface ablation stays paired.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from pim.environments.discworld import bench as dwb
from pim.environments.discworld.bench import EF
from pim.environments.discworld.tokens import UNK, FrameVocab, encode

ap = argparse.ArgumentParser()
ap.add_argument("--instance", default="dw-8ray")
ap.add_argument("--vocab", default="runs/interface_ablation/L-dw-8ray-tok-20m/vocab.npz")
ap.add_argument("--pool", type=int, default=4000)
ap.add_argument("--n", type=int, default=192)
ap.add_argument("--min-rays", type=int, default=2)
a = ap.parse_args()
root = REPO / "datasets/discworld" / a.instance
vocab = FrameVocab.load(REPO / a.vocab)
arr = dwb.bench_arrays(n=a.pool, target="full", basis_name="frustum", data_dir=root / "eval")
pre_f, post_f = arr["zones"].gt_unedited, arr["clean"][:, EF]
pre, post = encode(pre_f, vocab).astype(int), encode(post_f, vocab).astype(int)
tok = encode(arr["obs"][:, :EF], vocab).astype(int)
nray = (np.abs(pre_f - post_f) > 1e-6).sum(1)
ok = (pre != post) & (pre != UNK) & (post != UNK) & (tok != UNK).all(1) & (nray >= a.min_rays)
sel = np.where(ok)[0][: a.n]
i = np.arange(len(pre))
tele = np.linalg.norm(arr["pos"][i, EF, arr["edit_object"]] - arr["pos"][i, EF - 1, arr["edit_object"]], axis=-1)
out = {"instance": a.instance, "vocab": a.vocab, "pool": a.pool, "n": int(len(sel)), "min_rays": a.min_rays,
       "rule": "edited and unedited frames differ, both in the vocabulary, context in vocabulary, "
               f">= {a.min_rays} rays differ; the FIRST {a.n} such cases in pool order",
       "select": sel.tolist(),
       "stats": {"pool_scoreable": int(((pre != post) & (pre != UNK) & (post != UNK)).sum()),
                 "pool_identical_frame": int((pre == post).sum()),
                 "pool_ge_min_rays": int(ok.sum()),
                 "teleport_mean_selected": float(tele[sel].mean()),
                 "teleport_mean_pool": float(tele.mean()),
                 "rays_changed_selected": {int(k): int(v) for k, v in zip(*np.unique(nray[sel], return_counts=True))}}}
p = root / "edits_selection.json"
p.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "select"}, indent=1))
print(f"-> {p.relative_to(REPO)}  ({len(sel)} cases)")
