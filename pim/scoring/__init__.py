"""pim.scoring — the canonical scorer behind ``notebooks/master_eval.ipynb`` (2026-09-19).

The notebook keeps the run scan call, the SETTINGS (every knob, in one visible place) and the
version rule; everything that was wiring, schema or bookkeeping in its cells lives here, moved
verbatim. No metric math: every number is still a call into pim.probes / pim.editors /
pim.metrics / pim.environments.

    runs        which runs are scored (scan_runs; PIM_ONLY_RUNS / PIM_SKIP_TOPICS; still-training guard)
    blocks      which probe-target blocks a run gets, and the scores.json block schema (probe_block)
    discworld   score_discworld (frame models), score_discworld_tokens (frames as tokens)
    othello     score_othello
    baselines   the two decodability floors per (instance, architecture) -> runs/_baselines/
    driver      score_all: score what is missing or stale, ADD blocks / editors a current file lacks
    summary     the per-run human-readable headline
"""
from pim.scoring.baselines import score_all_baselines
from pim.scoring.driver import score_all
from pim.scoring.runs import scan_runs
from pim.scoring.summary import print_summaries

__all__ = ["scan_runs", "score_all_baselines", "score_all", "print_summaries"]
