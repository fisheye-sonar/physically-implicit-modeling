"""pim.environments.discworld — the discworld environment: simulation, rendering, data.

One environment *class*; a specific configuration of it (noise levels, object counts,
seed/index laws for every split) is an environment *instance*, described by an
``instance.json`` manifest inside its dataset directory.

Module map (a file per concern, ordered by how data flows):

    config.py         SimConfig — every physical and rendering parameter
    sim.py            ground-truth dynamics: Scene, simulate, visibility, frustum geometry
    renderer.py       analytic 1D ray-cast -> observation (the canonical channel)
    frustum.py        world -> frustum-basis coordinate change (probe bases)
    dataset.py        HDF5 generation for train/val/test splits
    edits_dataset.py  HDF5 generation for the mid-sequence-teleport edits split
    loading.py        reading dataset directories back: Dataset / EditsData / DatasetBundle
    dataloader.py     torch Dataset/DataLoader wrappers (lazy HDF5 and in-memory)
    bench.py          the editability bench: warmed model states + edit zones + targets

Opt-in extensions, OFF by default and pinned bit-identical to the defaults by tests —
present because the canonical renderer/dataset path gates on them, not because any
canonical number uses them:

    soft_render.py    differentiable rendering (soft_edge / shading / psf / occlusion)
    render2d.py       omniscient 2D raster channel (SimConfig.omni2d)

Demo-supporting only (used by scripts/demos/, never by analyses):

    viz.py            2D animation + observation waterfall drawing
    interactive.py    stateful step-able world for keyboard/driver demos
"""

from pim.environments.discworld.config import SimConfig
from pim.environments.discworld.dataset import (
    DatasetConfig,
    generate_dataset,
    load_sample,
    reconstruct_clean_obs,
)
from pim.environments.discworld.edits_dataset import EditDatasetConfig, generate_edits_dataset
from pim.environments.discworld.loading import (
    Dataset,
    DatasetBundle,
    EditsData,
    load_dataset,
    make_test_loader,
)
from pim.environments.discworld.renderer import render_frame, render_scene
from pim.environments.discworld.sim import (
    OBJECT_COLORS,
    Scene,
    compute_visibility,
    frustum_half_width,
    simulate,
)

__all__ = [
    "SimConfig",
    "Scene",
    "simulate",
    "compute_visibility",
    "frustum_half_width",
    "OBJECT_COLORS",
    "render_frame",
    "render_scene",
    "DatasetConfig",
    "generate_dataset",
    "load_sample",
    "reconstruct_clean_obs",
    "EditDatasetConfig",
    "generate_edits_dataset",
    "Dataset",
    "EditsData",
    "DatasetBundle",
    "load_dataset",
    "make_test_loader",
]
