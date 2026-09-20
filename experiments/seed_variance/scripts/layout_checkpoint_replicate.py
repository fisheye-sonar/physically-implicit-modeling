"""MOVED 2026-09-19 → ``scripts/layout_checkpoint_replicate.py`` (the canonical copy).

This forwarder keeps the old path working for ``scripts/drivers/replicate.sh`` while the paper_ci
queue is live (a driver must not be edited while bash is executing it — GOTCHAS 2026-09-08). Once the
queue has drained: point replicate.sh stage B at ``scripts/layout_checkpoint_replicate.py`` and
delete this file.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "layout_checkpoint_replicate.py"), run_name="__main__")
