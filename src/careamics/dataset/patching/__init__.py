"""Patching strategies and factory for the next-generation dataset."""

__all__ = [
    "FixedPatching",
    "FixedRandomPatching",
    "PatchSpecs",
    "Patching",
    "RandomPatching",
    "RegionSpecs",
    "SequentialPatching",
    "StratifiedPatching",
    "TileSpecs",
    "TiledPatching",
    "WholeSamplePatching",
    "create_patching",
    "is_tile_specs",
]

from .fixed_patching import FixedPatching
from .patch_specs import RegionSpecs, TileSpecs, is_tile_specs
from .patching import Patching, PatchSpecs
from .patching_factory import create_patching
from .random_patching import FixedRandomPatching, RandomPatching
from .sequential_patching import SequentialPatching
from .stratified_patching import StratifiedPatching
from .tiled_patching import TiledPatching
from .whole_sample_patching import WholeSamplePatching
