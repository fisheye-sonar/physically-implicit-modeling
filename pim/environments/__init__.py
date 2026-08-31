"""pim.environments — the worlds models are trained in.

Two environment classes: ``discworld`` (continuous 2D physics observed through a 1D
ray-cast; regression) and ``othello`` (Li et al.'s synthetic Othello move sequences;
next-token classification). An environment *instance* is one class at one fixed
configuration, packaged with the data for every split it defines — see the
``instance.json`` manifest inside each ``datasets/<class>/<instance>/`` directory.

Subpackages are imported explicitly (``from pim.environments import discworld``), not
re-exported here: the two environments share no symbols, and the point of the layout is
that code says which world it is talking about.
"""
