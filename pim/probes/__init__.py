"""pim.probes — the canonical probes: how state is read out of a model.

Two workhorses, one file each, sharing one verified body (``base.py``):

    linear.py     LINEAR — one affine map; closed-form lstsq fit (regression).
    mlp.py        MLP-128 — Li et al.'s §3.2 shape (one hidden layer, width 128),
                  plus the MLP ≥ linear tripwire (`check_probe_sanity`).

One non-default probe, for the code-size question and the multi-probe editor:

    nullspace.py  iterative nullspace-projection cascade of orthogonal linear probes.

Support:

    base.py       `WorldStateProbe` (the shared module), `collect_residuals`, `fit_probe`.
    cache.py      fingerprinted, provenance-verified probe cache.

Probe fits are ALWAYS held out by sequence, never by frame (consecutive frames are
near-duplicates; a row split leaks them and inflates every number).
"""

from pim.probes.base import WorldStateProbe, collect_residuals, fit_probe
from pim.probes.cache import ProbeCache, fingerprint
from pim.probes.linear import fit_linear
from pim.probes.mlp import CANONICAL_HIDDEN, ProbeSanityError, check_probe_sanity, fit_mlp
from pim.probes.nullspace import NullspaceCascade, fit_nullspace_cascade

__all__ = [
    "WorldStateProbe",
    "collect_residuals",
    "fit_probe",
    "fit_linear",
    "fit_mlp",
    "CANONICAL_HIDDEN",
    "ProbeSanityError",
    "check_probe_sanity",
    "NullspaceCascade",
    "fit_nullspace_cascade",
    "ProbeCache",
    "fingerprint",
]
