"""pim.simulator — the physical environment."""

from pim.simulator.config import SimConfig
from pim.simulator.sim import Scene, simulate, compute_visibility, frustum_half_width, OBJECT_COLORS
from pim.simulator.renderer import render_frame, render_scene
from pim.simulator.viz import animate_scene, save_animation, make_waterfall
from pim.simulator.dataset import DatasetConfig, generate_dataset, load_sample
from pim.simulator.edits_dataset import EditDatasetConfig, generate_edits_dataset

__all__ = [
    "SimConfig",
    "Scene",
    "simulate",
    "compute_visibility",
    "frustum_half_width",
    "OBJECT_COLORS",
    "render_frame",
    "render_scene",
    "animate_scene",
    "save_animation",
    "make_waterfall",
    "DatasetConfig",
    "generate_dataset",
    "load_sample",
    "EditDatasetConfig",
    "generate_edits_dataset",
]
