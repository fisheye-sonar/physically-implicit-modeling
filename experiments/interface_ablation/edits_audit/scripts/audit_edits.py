"""Audit the dw-8ray edit set as the TOKEN model sees it: how many cases actually change the
target frame, and by how much. Reference-only (the two clean renders), no model."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from pim.environments.discworld import bench as dwb
from pim.environments.discworld.bench import EF
from pim.environments.discworld.tokens import UNK, FrameVocab, encode

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
root = REPO / "datasets/discworld/dw-8ray"
vocab = FrameVocab.load(REPO / "runs/interface_ablation/L-dw-8ray-tok-20m/vocab.npz")
a = dwb.bench_arrays(n=N, target="full", basis_name="frustum", data_dir=root / "eval")
pre_f, post_f = a["zones"].gt_unedited, a["clean"][:, EF]
pre, post = encode(pre_f, vocab).astype(int), encode(post_f, vocab).astype(int)
tok = encode(a["obs"][:, :EF], vocab).astype(int)
same_tok = pre == post
unk = (pre == UNK) | (post == UNK) | (tok == UNK).any(1)
keep = ~same_tok & ~unk
d = np.abs(pre_f - post_f)
n_rays = (d > 1e-6).sum(1)                        # rays whose value changes at all
maxd = d.max(1)
tele = np.linalg.norm(a["pos"][np.arange(len(pre)), EF, a["edit_object"]]
                      - a["pos"][np.arange(len(pre)), EF - 1, a["edit_object"]], axis=-1)
rep = {"n": int(N), "identical_target_frame": int(same_tok.sum()), "unk": int(unk.sum()),
       "scoreable": int(keep.sum()), "frac_scoreable": float(keep.mean()),
       "first192_scoreable": int(keep[:192].sum()),
       "rays_changed_hist": {int(k): int(v) for k, v in zip(*np.unique(n_rays, return_counts=True))},
       "rays_changed_hist_scoreable": {int(k): int(v) for k, v in zip(*np.unique(n_rays[keep], return_counts=True))},
       "teleport_mean_all": float(tele.mean()), "teleport_mean_scoreable": float(tele[keep].mean()),
       "teleport_mean_identical": float(tele[same_tok].mean()),
       "maxdiff_quantiles_scoreable": [float(q) for q in np.quantile(maxd[keep], [.1, .25, .5, .75, .9])],
       "n_with_ge2_rays": int((keep & (n_rays >= 2)).sum()), "n_with_ge3_rays": int((keep & (n_rays >= 3)).sum())}
print(json.dumps(rep, indent=1))
(Path(__file__).resolve().parents[1] / "scores" / "audit.json").write_text(json.dumps(rep, indent=1))
