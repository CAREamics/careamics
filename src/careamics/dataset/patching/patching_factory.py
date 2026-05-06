"""Patching strategy factory."""

from collections.abc import Sequence

from careamics.config.data.data_config import PatchingConfig
from careamics.config.support.supported_patching import (
    SupportedPatching,
)

from .patching import Patching
from .random_patching import FixedRandomPatching, RandomPatching
from .stratified_patching import StratifiedPatching
from .tiled_patching import TiledPatching
from .whole_sample_patching import WholeSamplePatching


def create_patching(
    data_shapes: list[Sequence[int]], patching_config: PatchingConfig
) -> Patching:
    """Factory function to create a patching strategy based on the provided config.

    Parameters
    ----------
    data_shapes : list of Sequence of int
        The shapes of the data stacks to be patched.
    patching_config : PatchingConfig
        The configuration for the desired patching.

    Returns
    -------
    Patching
        An instance of the specified patching.
    """
    patch_class = None
    match patching_config.name:
        case SupportedPatching.RANDOM:
            patch_class = RandomPatching
        case SupportedPatching.STRATIFIED:
            patch_class = StratifiedPatching
        case SupportedPatching.FIXED_RANDOM:
            patch_class = FixedRandomPatching
        case SupportedPatching.TILED:
            patch_class = TiledPatching
        case SupportedPatching.WHOLE:
            patch_class = WholeSamplePatching
        case _:
            raise ValueError(f"Unsupported patching: {patching_config.name}")

    # remove `name` to match the class signatures
    # tiling requires `tile_size` instead of `patch_size`, hence the aliasing
    return patch_class(
        data_shapes=data_shapes, **patching_config.model_dump(exclude={"name"})
    )
