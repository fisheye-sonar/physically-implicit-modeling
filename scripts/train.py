#!/usr/bin/env python3
"""Canonical training entry point: one command, any (environment × architecture) cell.

    python scripts/train.py --env discworld --arch transformer_l \
        --topic initial_othello_comparison --run-name L-dw-20m --steps 780000

    python scripts/train.py --env othello --arch transformer_l \
        --topic initial_othello_comparison --run-name L-oth-20m --steps 780000

Everything of substance lives in ``pim.training`` (the loop and recipe), the model
registry (``pim.models``), and the environment corpora — this file only wires an
argument list to those pieces and picks the run directory ``runs/<topic>/<name>/``.

``--limit N`` trains on a strict PREFIX of the pool (the data-scale axis).
``--smoke`` = a 200-step configuration for verifying the pipeline end to end.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Cap CPU thread pools BEFORE torch loads (OpenMP reads the env at library load; the
# loop is GPU-bound and 32 CPU threads buy nothing — measured 2026-08-24).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from pim.models import build as build_model  # noqa: E402
from pim.training import TrainConfig, discworld_source, othello_source, token_source, train  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=("discworld", "othello"))
    p.add_argument("--arch", required=True,
                   choices=("transformer_s", "transformer_l", "recurrent_l"),
                   help="task head (regression vs tokens) is implied by --env")
    p.add_argument("--topic", required=True,
                   help="runs/<topic>/<run-name>/ — the line of work this run belongs to")
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="train on the first N sequences of the pool (data-scale axis)")
    p.add_argument("--instance", default=None,
                   help="environment instance (discworld: dw-pn04 | dw-noiseless | dw-8ray | dw-blink; "
                        "othello: oth-uniform | oth-noflip | oth-adjacent). Default: the env's canonical instance.")
    # the canonical recipe; override only deliberately
    p.add_argument("--repr", choices=("frames", "tokens"), default="frames",
                   help="discworld only. frames = the float observation into the regression head "
                        "(canonical). tokens = every frame is ONE token of the instance's frame "
                        "vocabulary (datasets/discworld/<instance>/tokens/, built by "
                        "scripts/make_discworld_tokens.py) into the Othello token model: embedding "
                        "in, softmax over frames + cross-entropy out — the interface ablation")
    p.add_argument("--objective", choices=("ce", "mse_onehot"), default="ce",
                   help="Othello only: cross-entropy (canonical) or MSE against the one-hot "
                        "next move — the head then emits raw probability estimates "
                        "(model output_kind='raw', read as-is by every scorer)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--ckpt-base", type=int, default=1_000)
    p.add_argument("--val-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="tiny cadence overrides for an end-to-end pipeline check")
    return p.parse_args()


def main() -> None:
    a = _parse()
    if a.smoke:
        a.warmup_steps = min(a.warmup_steps, 20)
        a.val_every = min(a.val_every, 50)
        a.ckpt_base = min(a.ckpt_base, 64)
    cfg = TrainConfig(steps=a.steps, batch_size=a.batch_size, lr=a.lr,
                      weight_decay=a.weight_decay, grad_clip=a.grad_clip,
                      lr_schedule=a.lr_schedule, warmup_steps=a.warmup_steps,
                      ckpt_base=a.ckpt_base, val_every=a.val_every, seed=a.seed)

    if a.env == "discworld" and a.repr == "tokens":
        if a.arch != "transformer_l":
            raise SystemExit("--repr tokens is implemented for transformer_l only")
        from pim.environments.discworld.tokens import load_tokens

        inst = a.instance or "dw-8ray"
        tdir = _REPO / "datasets" / "discworld" / inst / "tokens"
        tok, ln, vocab, tmeta = load_tokens(tdir)
        arch = "transformer_l_tokens"      # the Othello model, unchanged
        mc = {"vocab": int(vocab.size), "block_size": int(tmeta["n_frames"]) - 1}
        if a.objective != "ce":
            mc = {**mc, "output_kind": "raw"}
        source = token_source(tok, ln, block=mc["block_size"], env="discworld",
                              batch_size=a.batch_size, seed=a.seed, device=DEV,
                              limit=a.limit, objective=a.objective,
                              meta={"instance": inst, "repr": "tokens",
                                    "corpus": str(tdir / "train.i16"),
                                    "vocab": str(tdir / "vocab.npz"),
                                    "vocab_size": int(vocab.size)})
        run_dir = _REPO / "runs" / a.topic / a.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tdir / "vocab.npz", run_dir / "vocab.npz")   # the run stays self-contained
    elif a.env == "discworld":
        if a.objective != "ce":
            raise SystemExit("--objective applies to Othello; discworld trains with MSE on the next frame")
        from pim.environments.discworld import bigcorpus as bc

        bc.use_instance(a.instance or "dw-pn04")
        arch = a.arch  # regression head
        mc = ({"obs_res": bc.OBS_RES, "block_size": bc.FRAMES - 1}
              if a.arch == "transformer_l" else
              # Recurrent-L: 4 x 1024 GRU ~= Transformer-L's 25.4M params (2026-09-02)
              {"input_dim": bc.OBS_RES, "d_model": 1024, "n_layers": 4, "dropout": 0.1}
              if a.arch == "recurrent_l" else
              {"input_dim": bc.OBS_RES, "d_model": 256, "n_layers": 4,
               "n_heads": 4, "mlp_ratio": 4.0, "window": 16})
        source = discworld_source(bc.open_obs("r"), n_total=bc.N_TOTAL,
                                  batch_size=a.batch_size, seed=a.seed, device=DEV,
                                  limit=a.limit,
                                  meta={"instance": bc.INSTANCE, "corpus": str(bc.OUT)})
    else:
        from pim.environments.othello import corpus as oc

        if a.arch == "recurrent_l":
            raise SystemExit("recurrent_l has no token head yet (discworld only)")
        arch = a.arch + "_tokens"
        inst = a.instance or "oth-uniform"       # oth-noflip = the same generator, no recolouring
        paths = oc.build(a.limit or oc.LADDER["D"], only=("train",), instance=inst)
        tok, ln = oc.load(paths["train"])
        # vocabulary and block length come from the corpus itself, cross-checked against
        # the one definition in pim.environments.othello.data (61 tokens, block 59)
        from pim.environments.othello.data import T_MODEL, VOCAB
        vocab, block = int(tok.max()) + 1, int(tok.shape[1]) - 1
        assert (vocab, block) == (VOCAB, T_MODEL), (vocab, block, VOCAB, T_MODEL)
        mc = ({"vocab": vocab, "block_size": block}
              if a.arch == "transformer_l" else
              {"input_dim": 128, "d_model": 256, "n_layers": 4, "n_heads": 4,
               "mlp_ratio": 4.0, "window": 16, "vocab": vocab})
        if a.objective != "ce":
            if a.arch != "transformer_l":
                raise SystemExit("--objective mse_onehot is implemented for transformer_l only")
            mc = {**mc, "output_kind": "raw"}     # the head's outputs ARE the estimates
        source = othello_source(tok, ln, batch_size=a.batch_size, seed=a.seed,
                                device=DEV, limit=a.limit, objective=a.objective,
                                meta={"instance": inst, **oc.rules_of(inst),
                                      "corpus": str(paths["train"])})

    model = build_model(arch, mc)
    run_dir = _REPO / "runs" / a.topic / a.run_name
    train(model, source, cfg, run_dir, arch=arch, model_config=mc, device=DEV)


if __name__ == "__main__":
    main()
