"""pim.environments — the worlds models are trained in.

Two environment classes: ``discworld`` (continuous 2D physics observed through a 1D
ray-cast; regression) and ``othello`` (Li et al.'s synthetic Othello move sequences;
next-token classification). An environment *instance* is one class at one fixed
configuration, packaged with the data for every split it defines under
``datasets/<class>/<instance>/`` (``train/corpus.json`` + split ``config_json`` are the
contracts; ``instance.json`` is a hand-written summary code never reads). Where every
split lives is ``layout.py`` — layout v2 (2026-09-10): ``train/ probe/ eval/ edits/v1/``
role directories, ``layout.json`` marker, logical probe-cache keys. No other module
spells a path under ``datasets/``.

Subpackages are imported explicitly (``from pim.environments import discworld``), not
re-exported here: the two environments share no symbols, and the point of the layout is
that code says which world it is talking about.
"""
