"""Explicit model registry: name → builder, and the one checkpoint loader.

Replaces the old ``pim/world_models/loader.py`` key-sniffing dispatch. A new-scheme
checkpoint names its architecture explicitly (``ckpt["arch"]``, written by
``pim.training``); the two legacy formats still in service — the canonical BIG20M runs
and the archived S runs — are recognised by the *documented* rules below, not by
guessing. Anything else (GRU/RSSM/DiT checkpoints) is out of scope: recover the old
loader from the ``pre-cleanup-2026-08`` tag if one ever needs to be opened again.

Legacy recognition rules (exact, in order):
  1. ``model_config`` has ``obs_res`` and ``block_size``  → ``transformer_l``
     (saved by ``othello_arch/train.py`` / ``discworld_scale/train.py``).
  2. top-level ``vocab`` key and ``model_config`` has ``window``  →
     ``transformer_s_tokens`` (saved by ``ours_on_othello/train.py``).
  3. ``model_config`` has ``window`` and ``input_dim``  → ``transformer_s``
     (saved by ``scripts/train_transformer.py``).
  4. every state-dict key starts with ``gpt.`` and ``gpt.tok_emb.weight`` exists →
     ``transformer_l_tokens`` (a bare retrained minGPT).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from pim.models.recurrent import RecurrentConfig, RecurrentL
from pim.models.transformer_l import TransformerL, TransformerLTokens
from pim.models.transformer_s import ModelConfig as SConfig
from pim.models.transformer_s import TransformerS, TransformerSTokens


def _build_s(cfg: dict):
    return TransformerS(SConfig(**cfg))


def _build_s_tokens(cfg: dict):
    cfg = dict(cfg)
    vocab = cfg.pop("vocab", 61)
    return TransformerSTokens(SConfig(**cfg), vocab=vocab)


def _build_l(cfg: dict):
    keep = {k: cfg[k] for k in ("obs_res", "block_size", "n_layer", "n_head",
                                "n_embd", "dropout") if k in cfg}
    return TransformerL(**keep)


def _build_l_tokens(cfg: dict):
    keep = {k: cfg[k] for k in ("vocab", "block_size", "n_layer", "n_head",
                                "n_embd", "dropout") if k in cfg}
    return TransformerLTokens(**keep)


def _build_recurrent(cfg: dict):
    return RecurrentL(RecurrentConfig(**cfg))


BUILDERS = {
    "recurrent_l": _build_recurrent,
    "transformer_s": _build_s,
    "transformer_s_tokens": _build_s_tokens,
    "transformer_l": _build_l,
    "transformer_l_tokens": _build_l_tokens,
}


def build(arch: str, model_config: dict):
    """Instantiate a registered architecture from its config dict."""
    if arch not in BUILDERS:
        raise KeyError(f"unknown arch {arch!r}; registered: {sorted(BUILDERS)}")
    return BUILDERS[arch](model_config)


@dataclass
class CheckpointInfo:
    arch: str
    val_loss: float
    model_config: dict
    train_config: dict
    run_dir: Path
    ckpt: dict  # the raw checkpoint dict, for fields not modelled here


def _infer_arch(ckpt: dict) -> str:
    if "arch" in ckpt:
        return ckpt["arch"]
    mc = ckpt.get("model_config") or {}
    sd = ckpt.get("model_state") or {}
    if "obs_res" in mc and "block_size" in mc:
        return "transformer_l"
    if "vocab" in ckpt and "window" in mc:
        return "transformer_s_tokens"
    if "window" in mc and "input_dim" in mc:
        return "transformer_s"
    if sd and all(k.startswith("gpt.") for k in sd) and "gpt.tok_emb.weight" in sd:
        return "transformer_l_tokens"
    raise ValueError(
        f"cannot identify architecture: no 'arch' key and no legacy rule matches "
        f"(model_config keys: {sorted(mc)})"
    )


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Load any registered checkpoint → (model in eval mode, CheckpointInfo).

    The returned model has ``requires_grad_(False)`` — analysis code that needs
    gradients w.r.t. *activations* (gradient steering) does, and should, take them
    on the activation tensors, never on the weights.
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = _infer_arch(ckpt)
    mc = dict(ckpt.get("model_config") or {})
    if arch == "transformer_s_tokens" and "vocab" in ckpt:
        mc["vocab"] = ckpt["vocab"]
    model = build(arch, mc).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, CheckpointInfo(
        arch=arch,
        val_loss=float(ckpt.get("val_loss", float("nan"))),
        model_config=mc,
        train_config=ckpt.get("train_config", {}),
        run_dir=path.parent,
        ckpt=ckpt,
    )


def load_run(run_dir: str | Path, device: str = "cpu", ckpt_name: str = "best_model.pt"):
    """Load a run directory's checkpoint plus its ``config.json`` if present."""
    run_dir = Path(run_dir)
    model, info = load_checkpoint(run_dir / ckpt_name, device=device)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        info.train_config = {**json.loads(cfg_path.read_text()), **info.train_config}
    return model, info
