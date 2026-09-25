"""READ-ONLY inventory of the in-scope runs: files + sizes, probe-cache classification, fingerprint check,
and the scorer's skip decision (driver.missing_blocks / missing_inverse against the notebook SETTINGS).
Writes only to the scratchpad."""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
from settings import REPO, load_settings  # noqa: E402

sys.path.insert(0, str(REPO))
from pim.models import load_checkpoint  # noqa: E402
from pim.probes.cache import fingerprint  # noqa: E402
from pim.probes.inverse import CATEGORICAL_STATE, INVERSE_EPOCHS, INVERSE_HIDDEN  # noqa: E402
from pim.environments.discworld.grid_target import categorical_target, snapped_target  # noqa: E402
from pim.scoring import driver  # noqa: E402
from pim.scoring.blocks import dw_bases_for, extra_targets_of  # noqa: E402
from pim.scoring.runs import training_complete  # noqa: E402

S, eval_version, EVB, EV0 = load_settings()

MAIN = {
    "oth-standard": "initial_othello_comparison/L-oth-20m",
    "oth-adjacent-flip": "adjacent_flip_ablation/L-oth-adjacent-flip-20m",
    "oth-adjacent-noflip": "adjacency_ablation/L-oth-adjacent-20m",
    "oth-standard-noflip": "flip_ablation/L-oth-noflip-20m",
    "rw-standard": "noise_ablation/L-dw-noiseless-20m",
    "rw-blink": "blink_ablation/L-dw-blink-20m",
    "rw-128ray": "ray_ablation/L-dw-128ray-20m",
    "rw-16ray": "ray_ablation/L-dw-16ray-20m",
    "rw-8ray": "ray_ablation/L-dw-8ray-20m",
    "rw-5ray": "ray_ablation/L-dw-5ray-20m",
    "rw-smooth": "smooth_ablation/L-dw-smooth-20m",
    "rw-obs5": "observer_ablation/L-dw-8ray-obs5-20m",
    "rw-8ray-tok": "interface_ablation/L-dw-8ray-tok-20m",
}
# brief's "paper scope" for categorical targets (everything else categorical = out of scope)
PAPER_CAT = {"appearance-fac", "appearance", "grid-16x8", "grid-6x5", "grid-10x3", "pos@appearance"}
# what the qualitative figure scripts load (discworld: CONT full/cartesian + CAT appearance-fac/frustum)
FIG_DW_RUNS = {"noise_ablation/L-dw-noiseless-20m", "blink_ablation/L-dw-blink-20m", "ray_ablation/L-dw-16ray-20m",
               "ray_ablation/L-dw-8ray-20m", "ray_ablation/L-dw-5ray-20m"}
FIG_OTH_RUNS = {MAIN[k] for k in ("oth-standard", "oth-adjacent-flip", "oth-adjacent-noflip", "oth-standard-noflip")}


def du(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                t += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return t


def run_dirs():
    out = []
    for tag, rk in MAIN.items():
        d = REPO / "runs" / rk
        out.append((tag, rk, d, False))
        for rp in sorted(d.parent.glob(f"{d.name}__seed*")):
            out.append((tag, f"{rk.split('/')[0]}/{rp.name}", rp, True))
    return out


def classify_dw(prov, inst, bases, extra, tokens_tag, cat_scope):
    kind = prov.get("kind")
    tgt = prov.get("target")
    basis = prov.get("basis")
    data_ok = prov.get("data") == f"discworld/{inst}"
    enc_ok = prov.get("encoder") == tokens_tag
    if not data_ok:
        return "dead", f"data key {prov.get('data')!r}"
    if not enc_ok:
        return "dead", f"encoder {prov.get('encoder')!r}"
    if kind is None:
        if int(prov.get("seed", 0)) != 0:
            return "dead", "probe-seed replicate (seed != 0)"
        if tgt == "full":
            if prov.get("n_seq") == 30000 and prov.get("split") == "probe_120k" and "epochs" not in prov:
                return ("req", f"fwd full/{basis}") if basis in bases else ("dead", f"fwd full/{basis} (basis not scored)")
            return "dead", f"fwd full other recipe n={prov.get('n_seq')} {prov.get('split')} ep={prov.get('epochs')}"
        if tgt == "pos":
            return "dead", "retired pos target"
        if categorical_target(tgt) is not None:
            if tgt in extra and basis == bases[0] and prov.get("n_seq") == 200000 and prov.get("epochs") == 50 \
                    and prov.get("split") == "probe_250k":
                return "req", f"fwd cat {tgt}"
            return "dead", f"fwd cat {tgt}/{basis} not requested"
        if snapped_target(tgt) is not None:
            if tgt in extra and prov.get("n_seq") == 30000:
                return "req", f"fwd snapped {tgt}"
            return "dead", f"snapped {tgt} not requested"
        return "dead", f"fwd {tgt}"
    if kind == "inverse_map":
        if prov.get("state") == CATEGORICAL_STATE:
            if tgt in extra and tgt in cat_scope and basis == bases[0] and prov.get("n_seq") == 200000:
                return "req", f"IM cat {tgt}"
            return "dead", f"IM cat {tgt} out of dw_cat_im scope"
        if tgt == "full" and prov.get("n_seq") == 30000 and basis in bases and int(prov.get("seed", 0)) == 0 \
                and prov.get("hidden") == INVERSE_HIDDEN and prov.get("epochs") == INVERSE_EPOCHS:
            return "req", f"IM full/{basis}"
        return "dead", f"IM other {tgt}/{basis} n={prov.get('n_seq')}"
    return "dead", f"kind {kind}"


def classify_oth(prov):
    kind = prov.get("kind")
    if kind == "othello_grid":
        ok = (prov.get("families") == ["linear", "mlp"] and prov.get("splits") == ["sequence"]
              and prov.get("n_seq") == 20000 and int(prov.get("seed", 0)) == 0)
        if ok and prov.get("targets") == ["mine"]:
            return "req", "grid mine"
        if ok and prov.get("targets") == ["mine_signed"]:
            return "req", "grid mine_signed"
        return "dead", f"grid {prov.get('targets')} {prov.get('families')} {prov.get('splits')} n={prov.get('n_seq')}"
    if kind == "inverse_map" and prov.get("target") == "mine-onehot" and prov.get("n_games") == 20000 \
            and int(prov.get("seed", 0)) == 0:
        return "req", "IM mine-onehot"
    return "dead", f"kind {kind} target {prov.get('target') or prov.get('targets')}"


def tier_of(cat, detail, env, rk_parent, block_is_paper):
    """paper = the brief's paper scope; fig = what the qualitative figures load."""
    return block_is_paper


def main():
    SCR_OUT = SCR / "inventory_runs.json"
    res = []
    fp_cache = {}
    for tag, rk, d, is_rep in run_dirs():
        cfg = json.loads((d / "config.json").read_text())
        env = cfg["data"]["env"]
        inst = cfg["data"]["instance"]
        arch = cfg["arch"]
        parent_rk = cfg.get("replicate", {}).get("of") if is_rep else rk
        r = {"tag": tag, "run": rk, "replicate": is_rep, "env": env, "instance": inst, "arch": arch,
             "replicate_cfg": cfg.get("replicate"), "corpus_path": cfg["data"].get("corpus")}
        # top-level entries
        ent = {}
        for p in sorted(d.iterdir()):
            ent[p.name + ("/" if p.is_dir() else "")] = du(p)
        r["entries"] = ent
        # best_model contents + fingerprint
        ck = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False, mmap=True)
        r["best_model_keys"] = sorted(ck.keys())
        r["best_model_val_loss"] = ck.get("val_loss")
        r["best_model_step"] = ck.get("step")
        del ck
        model, info = load_checkpoint(d / "best_model.pt", device="cpu")
        fp = fingerprint(model)
        r["fingerprint"] = fp
        r["n_layers"] = model.n_layers
        del model
        # ckpt dir detail
        if (d / "ckpt").is_dir():
            ck_files = sorted((d / "ckpt").glob("*.pt"))
            r["ckpt_n"] = len(ck_files)
            r["ckpt_names_head"] = [p.name for p in ck_files[:3]] + ["..."] + [p.name for p in ck_files[-3:]]
        # training_complete
        r["training_complete"] = training_complete(d, cfg)
        # scores
        sp = d / "scores.json"
        row = {"topic": rk.split("/")[0], "run": d.name, "dir": d, "arch": arch, "env": env, "instance": inst}
        if sp.exists():
            sc = json.loads(sp.read_text())
            r["eval_version"] = sc.get("eval_version")
            r["eval_version_wanted"] = eval_version(row)
            r["bases"] = list(sc.get("bases", {}))
            r["blocks_added"] = list(sc.get("blocks_added", {}))
            r["inverse_added"] = {k: v.get("version") for k, v in sc.get("inverse_added", {}).items()}
            r["has_prediction"] = "prediction" in sc
            r["prediction_readings"] = list(sc.get("prediction", {}).get("readings", {}) or {})
            r["probe_dir_field"] = sc.get("probe_dir")
            im_by_block = {}
            nn_by_block = {}
            for k, b in sc.get("bases", {}).items():
                im_by_block[k] = any(a.get("editor") == "IM" for a in b.get("arms", []))
                inv = b.get("inverse_map")
                nn_by_block[k] = (None if not isinstance(inv, dict) else (inv.get("nn_r2") is not None))
            if env == "othello":
                im_by_block["mine/theirs"] = any(a.get("editor") == "IM" for a in sc.get("arms", []))
                inv = sc.get("inverse_map")
                nn_by_block["mine/theirs(top)"] = (None if not isinstance(inv, dict) else inv.get("nn_r2") is not None)
            r["IM_by_block"] = im_by_block
            r["nn_r2_by_block"] = nn_by_block
            # scorer decisions under default env, and with the two opt-in catch-up flags
            for flag in (None, "PIM_ADD_CAT_IM", "PIM_ADD_NN_R2"):
                env_bak = dict(os.environ)
                if flag:
                    os.environ[flag] = "1"
                try:
                    mb = driver.missing_blocks(row, sc, S)
                    mi = driver.missing_inverse(row, sc, S)
                finally:
                    os.environ.clear(); os.environ.update(env_bak)
                r[f"decision[{flag or 'default'}]"] = {
                    "version_ok": sc.get("eval_version") == eval_version(row),
                    "missing_blocks": mb, "missing_inverse": mi,
                    "skip": sc.get("eval_version") == eval_version(row) and not mb and not mi}
        else:
            r["scores"] = None
        # probes
        pdir = d / "probes"
        probes = []
        if pdir.is_dir():
            bases = dw_bases_for(inst, S) if env == "discworld" else None
            extra = extra_targets_of(rk, S) if env == "discworld" else None
            cat_scope = set(S["dw_cat_im"]["targets"]) if inst in S["dw_cat_im"]["instances"] else set()
            tokens_tag = None
            if arch.endswith("_tokens") and env == "discworld":
                from pim.environments.discworld.token_bench import token_encoder
                from pim.environments.discworld.tokens import FrameVocab
                tokens_tag = token_encoder(FrameVocab.load(d / "vocab.npz"))[1]
                r["tokens_tag"] = tokens_tag
            for p in sorted(pdir.rglob("*")):
                if p.is_dir():
                    continue
                rel = str(p.relative_to(pdir))
                e = {"file": rel, "bytes": p.stat().st_size}
                if p.suffix == ".pt" and "/" not in rel:
                    try:
                        blob = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
                        prov = blob["provenance"]
                        del blob
                    except Exception as ex:  # noqa: BLE001
                        e["cat"], e["detail"] = "unreadable", repr(ex)[:80]
                        probes.append(e)
                        continue
                    e["prov"] = {k: v for k, v in prov.items()}
                    e["fp_ok"] = prov.get("model") == fp
                    if not e["fp_ok"]:
                        e["cat"], e["detail"] = "dead", f"fingerprint {prov.get('model')} != {fp}"
                    elif env == "discworld":
                        e["cat"], e["detail"] = classify_dw(prov, inst, bases, extra, tokens_tag, cat_scope)
                    else:
                        e["cat"], e["detail"] = classify_oth(prov)
                else:
                    e["cat"], e["detail"] = ("index" if rel == "INDEX.md" else "dead"), (
                        "INDEX.md (derived)" if rel == "INDEX.md" else "subdir/other")
                probes.append(e)
        r["probes"] = probes
        res.append(r)
        print(f"done {rk}  fp {fp}  probes {len(probes)}", flush=True)
    SCR_OUT.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", SCR_OUT)


if __name__ == "__main__":
    main()
