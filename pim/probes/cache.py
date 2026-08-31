"""Fingerprinted, provenance-verified probe cache.

Ported 2026-08-31 from ``othello_arch/editability.py``. The two rules it enforces exist
because each was violated once and produced wrong numbers:

1. **The model's weights are part of the key.** A cache key that omits them is how the
   random-init control was once served the trained model's probes (2026-08-21), and
   both then reported identical error.
2. **A hit is verified against the stored provenance before it is returned**, so a key
   collision or a hand-edited cache file raises instead of silently serving the wrong
   probe. Writes are atomic (``.partial`` → ``replace``): a killed run never leaves a
   half-written file that later reads as a hit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def fingerprint(model) -> str:
    """12 hex chars over every parameter — a model's identity, cheaply."""
    h = hashlib.blake2b(digest_size=6)
    for _, v in sorted(model.state_dict().items()):
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class ProbeCache:
    """One directory of cached probe fits, each stored with its full provenance."""

    #: bump to invalidate every cached probe after a change to probe fitting
    VERSION = 2  # v2 = the pim.probes port (MLP width 128 canonical)

    def __init__(self, cache_dir: str | Path) -> None:
        self.dir = Path(cache_dir)

    def key(self, model, **prov) -> tuple[str, dict]:
        """(filename, provenance). Every input that changes the fitted probe belongs in
        ``prov`` — target, n_seq, split, hidden, basis, seed, data path, …"""
        full = {"model": fingerprint(model),
                "span": int(getattr(model, "state_span", -1)),
                "v": self.VERSION, **prov}
        h = hashlib.blake2b(repr(sorted(full.items())).encode(), digest_size=8).hexdigest()
        return f"probes_{h}.pt", full

    def load(self, fname: str, prov: dict, device: str = "cpu"):
        """Return the cached probes, or None on a miss. Raises on provenance mismatch."""
        fpath = self.dir / fname
        if not fpath.exists():
            return None
        blob = torch.load(fpath, map_location=device, weights_only=False)
        if blob.get("provenance") != prov:
            raise RuntimeError(
                f"probe cache provenance mismatch at {fpath}\n"
                f"  on disk: {blob.get('provenance')}\n  wanted : {prov}\n"
                f"Delete the file to refit. This should be unreachable — the filename is "
                f"a hash of exactly this dict — so reaching it means the cache was "
                f"tampered with.")
        return blob["probes"]

    def store(self, fname: str, prov: dict, probes) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = (self.dir / fname).with_suffix(".pt.partial")
        torch.save({"provenance": prov, "probes": probes}, tmp)
        tmp.replace(self.dir / fname)  # atomic
