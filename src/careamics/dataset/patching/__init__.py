"""Patching strategies and factory for the next-generation dataset."""

__all__ = [
    "FixedPatching",
    "FixedRandomPatching",
    "PatchSpecs",
    "Patching",
    "PerChannelPatchSpecs",
    "RandomPatching",
    "RegionSpecs",
    "SequentialPatching",
    "StratifiedPatching",
    "TileSpecs",
    "TiledPatching",
    "WholeSamplePatching",
    "create_patching",
    "is_per_channel_specs",
    "is_tile_specs",
    "pack_patch_spec",
    "unpack_patch_spec",
]

from .fixed_patching import FixedPatching
from .patch_specs import (
    PerChannelPatchSpecs,
    RegionSpecs,
    TileSpecs,
    is_per_channel_specs,
    is_tile_specs,
    pack_patch_spec,
    unpack_patch_spec,
)
from .patching import Patching, PatchSpecs
from .patching_factory import create_patching
from .random_patching import FixedRandomPatching, RandomPatching
from .sequential_patching import SequentialPatching
from .stratified_patching import StratifiedPatching
from .tiled_patching import TiledPatching
from .whole_sample_patching import WholeSamplePatching
