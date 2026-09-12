#!/usr/bin/env python3
"""Canonical training entry point: one command, any (environment × architecture) cell.

    python scripts/train.py --env discworld --arch transformer_l \
        --topic initial_othello_comparison --run-name L-dw-20m --steps 780000

    python scripts/train.py --env othello --arch transformer_l \
        --topic initial_othello_comparison --run-name L-oth-20m --steps 780000

Everything of substance lives in ``pim.training`` (the loop and recipe), the model
registry (``pim.models``), and the environment corpora — this file only wires an
argument list to those pieces and picks the run directory ``runs/<topic>/<name>/``.

The INTERFACE is a parameter, not a property of the environment (2026-09-09):

    --repr frames    float observations through a linear encoder, a frame-regression head,
                     MSE on the next frame            (discworld's canonical setup)
    --repr tokens    integer ids through an embedding, a categorical head over the
                     vocabulary, and --objective:
                        ce          softmax + cross-entropy on the next token  (Othello's canonical setup)
                        mse_onehot  MSE against the one-hot next token; the head emits raw
                                    estimates (model output_kind "raw", read as-is by every scorer)

Othello has only a token corpus (a move IS a token), so it always trains with ``--repr
tokens``; discworld has both — its frame corpus, and a frame VOCABULARY on the noiseless
8-ray instance (``datasets/discworld/<instance>/tokens/``, ``scripts/make_discworld_tokens.py``).
Every combination goes through the same model class, loop, objectives and scorers.

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
from pim.training import TrainConfig, discworld_source, token_source, train  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CANONICAL_INSTANCE = {"discworld": "dw-pn04", "othello": "oth-uniform"}


def _parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, choices=("discworld", "othello"))
    p.add_argument("--arch", required=True,
                   choices=("transformer_s", "transformer_l", "recurrent_l"),
                   help="the body; the interface comes from --repr / --objective")
    p.add_argument("--topic", required=True,
                   help="runs/<topic>/<run-name>/ — the line of work this run belongs to")
    p.add_argument("--run-name", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="train on the first N sequences of the pool (data-scale axis)")
    p.add_argument("--instance", default=None,
                   help="environment instance (discworld: dw-pn04 | dw-noiseless | dw-8ray | dw-blink; "
                        "othello: oth-uniform | oth-noflip | oth-adjacent | oth-adjacent-flip). "
                        "Default: the env's canonical instance.")
    # the interface — see the module docstring
    p.add_argument("--repr", choices=("frames", "tokens"), default=None,
                   help="frames = float observations, linear encoder, frame-regression head, MSE "
                        "(discworld default). tokens = ids, embedding, categorical head over the "
                        "vocabulary (Othello always; discworld via the instance's frame vocabulary).")
    p.add_argument("--objective", choices=("ce", "mse_onehot"), default="ce",
                   help="categorical head only: cross-entropy (canonical) or MSE against the "
                        "one-hot next token (the head then emits raw estimates, output_kind='raw')")
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
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --topic/--run-name from its ckpt/latest.pt up to --steps "
                        "(which may exceed the original: a finished run can be extended)")
    p.add_argument("--dropout", type=float, default=None,
                   help="transformer_l only: override the body's dropout (class default 0.1 on the "
                        "embedding, attention and residual paths); 0 disables it")
    return p.parse_args()


# ── the corpora: what each environment can feed ──────────────────────────────


def _discworld_frames(inst: str):
    """The 410 GB frame memmap → (obs memmap, n_total, obs_dim, block, meta)."""
    from pim.environments.discworld import bigcorpus as bc

    bc.use_instance(inst)
    return (bc.open_obs("r"), bc.N_TOTAL, bc.OBS_RES, bc.FRAMES - 1,
            {"instance": bc.INSTANCE, "corpus": str(bc.OUT)})


def _discworld_tokens(inst: str, run_dir: Path):
    """The instance's frame vocabulary → (tok, ln, vocab_size, block, meta). Copies
    vocab.npz into the run dir so the run stays self-contained."""
    from pim.environments.discworld.tokens import load_tokens
    from pim.environments.layout import tokens_dir

    tdir = tokens_dir(inst)
    tok, ln, vocab, tmeta = load_tokens(tdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(tdir / "vocab.npz", run_dir / "vocab.npz")
    return (tok, ln, int(vocab.size), int(tmeta["n_frames"]) - 1,
            {"instance": inst, "repr": "tokens", "corpus": str(tdir / "train.i16"),
             "vocab": str(tdir / "vocab.npz"), "vocab_size": int(vocab.size)})


def _othello_tokens(inst: str, limit):
    """Othello's move corpus → (tok, ln, vocab_size, block, meta); the vocabulary and block
    come from the corpus itself, cross-checked against the one definition in
    ``pim.environments.othello.data`` (61 tokens, block 59)."""
    from pim.environments.othello import corpus as oc
    from pim.environments.othello.data import T_MODEL, VOCAB

    paths = oc.build(limit or oc.LADDER["D"], only=("train",), instance=inst)
    tok, ln = oc.load(paths["train"])
    vocab, block = int(tok.max()) + 1, int(tok.shape[1]) - 1
    assert (vocab, block) == (VOCAB, T_MODEL), (vocab, block, VOCAB, T_MODEL)
    return tok, ln, vocab, block, {"instance": inst, **oc.rules_of(inst),
                                   "corpus": str(paths["train"])}


# ── the model for an interface ───────────────────────────────────────────────


def _model_config(arch: str, repr_: str, dim: int, block: int, objective: str,
                  dropout: float | None = None) -> tuple[str, dict]:
    """(registry name, model_config) for a body under an interface. ``dim`` is the
    observation dimension (frames) or the vocabulary size (tokens). ``dropout`` (2026-09-11)
    overrides Transformer-L's class default when given; it is recorded in the config."""
    if arch != "transformer_l" and dropout is not None:
        raise SystemExit("--dropout is implemented for transformer_l only")
    if repr_ == "frames":
        if arch == "transformer_l":
            return arch, {"obs_res": dim, "block_size": block, **({"dropout": dropout} if dropout is not None else {})}
        if arch == "recurrent_l":     # 4 x 1024 GRU ~= Transformer-L's 25.4M params (2026-09-02)
            return arch, {"input_dim": dim, "d_model": 1024, "n_layers": 4, "dropout": 0.1}
        return arch, {"input_dim": dim, "d_model": 256, "n_layers": 4, "n_heads": 4,
                      "mlp_ratio": 4.0, "window": 16}
    if arch == "recurrent_l":
        raise SystemExit("recurrent_l has no categorical head (frames only)")
    if arch == "transformer_l":
        mc = {"vocab": dim, "block_size": block, **({"dropout": dropout} if dropout is not None else {})}
        if objective != "ce":
            mc["output_kind"] = "raw"           # the head's outputs ARE the estimates
        return "transformer_l_tokens", mc
    if objective != "ce":
        raise SystemExit("--objective mse_onehot is implemented for transformer_l only")
    return "transformer_s_tokens", {"input_dim": 128, "d_model": 256, "n_layers": 4,
                                    "n_heads": 4, "mlp_ratio": 4.0, "window": 16, "vocab": dim}


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
    inst = a.instance or CANONICAL_INSTANCE[a.env]
    run_dir = _REPO / "runs" / a.topic / a.run_name
    # Othello has no frame corpus: a move is a token. Discworld defaults to its frames.
    repr_ = a.repr or ("tokens" if a.env == "othello" else "frames")
    if a.env == "othello" and repr_ == "frames":
        raise SystemExit("Othello has no frame corpus — its observations are move tokens (--repr tokens)")
    if repr_ == "frames" and a.objective != "ce":
        raise SystemExit("--objective selects the loss on a categorical head; a frame-regression "
                         "model trains with MSE on the next frame")
    if a.env == "discworld" and repr_ == "tokens" and a.arch != "transformer_l":
        raise SystemExit("discworld --repr tokens is implemented for transformer_l only")

    if repr_ == "frames":
        obs, n_total, dim, block, meta = _discworld_frames(inst)
        arch, mc = _model_config(a.arch, repr_, dim, block, a.objective, a.dropout)
        source = discworld_source(obs, n_total=n_total, batch_size=a.batch_size, seed=a.seed,
                                  device=DEV, limit=a.limit, meta=meta)
    else:
        tok, ln, dim, block, meta = (_discworld_tokens(inst, run_dir) if a.env == "discworld"
                                     else _othello_tokens(inst, a.limit))
        arch, mc = _model_config(a.arch, repr_, dim, block, a.objective, a.dropout)
        source = token_source(tok, ln, block=block, env=a.env, batch_size=a.batch_size,
                              seed=a.seed, device=DEV, limit=a.limit, objective=a.objective,
                              meta=meta)

    model = build_model(arch, mc)
    train(model, source, cfg, run_dir, arch=arch, model_config=mc, device=DEV, resume=a.resume)


if __name__ == "__main__":
    main()
