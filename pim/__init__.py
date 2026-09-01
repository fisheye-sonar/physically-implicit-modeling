"""physically-implicit-modeling (pim) — the canonical core.

The project studies **world-model editability**: probes can read object state out of a
trained sequence model, but can a write to that read-out change what the model
generates? Two environments answer differently — Othello (editable at scale) and
discworld (not) — and everything here exists to make that comparison airtight: the
architectures, probes, editors, metrics, and training recipe are IDENTICAL across
environments up to the input/output projection and the loss.

Five packages, strict roles (index of every canonical object: ``research/REGISTRY.md``):

    environments/   the worlds — discworld (sim, rendering, data, bench) and othello
                    (vendored generator, corpus, bench, arms); an *instance* packages
                    one configuration with all its splits (instance.json)
    models/         Transformer-S and Transformer-L, each with regression AND token
                    heads; protocol.py documents THE surface everything drives
    probes/         LIN + MLP-128 (+ the nullspace cascade); fits held out by
                    sequence; fingerprinted caches
    editors/        PI / ND / GS workhorses + nullspace + two oracle editors
    metrics/        decodability (Probe Skill), discworld and Othello editability —
                    arrays in, numbers out, never imports matplotlib
    training/       ONE loop, two objectives; defaults ARE the canonical recipe
    figures/        theme + the canonical waterfall + scaling panels

Canonical scores come from ``notebooks/master_eval.ipynb`` into each run's
``scores.json``; the pre-2026-08-31 tree lives at the ``pre-cleanup-2026-08`` git tag.
"""
