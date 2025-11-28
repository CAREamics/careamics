"""Patching strategy factory."""

from collections.abc import Sequence

from careamics.config.data.ng_data_config import PatchingConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)

from .patching_strategy_protocol import PatchingStrategy
from .random_patching import FixedRandomPatchingStrategy, RandomPatchingStrategy
from .tiling_strategy import TilingStrategy
from .whole_sample import WholeSamplePatchingStrategy


def create_patching_strategy(
    data_shapes: list[Sequence[int]], patching_config: PatchingConfig
) -> PatchingStrategy:
    """Factory function to create a patching strategy based on the provided config.

    Parameters
    ----------
    data_shapes : list of Sequence of int
        The shapes of the data stacks to be patched.
    patching_config: PatchingConfig
        The configuration for the desired patching strategy.

    Returns
    -------
    PatchingStrategy
        An instance of the specified patching strategy.
    """
    patch_class = None
    match patching_config.name:
        case SupportedPatchingStrategy.RANDOM:
            patch_class = RandomPatchingStrategy
        case SupportedPatchingStrategy.FIXED_RANDOM:
            patch_class = FixedRandomPatchingStrategy
        case SupportedPatchingStrategy.TILED:
            patch_class = TilingStrategy
        case SupportedPatchingStrategy.WHOLE:
            patch_class = WholeSamplePatchingStrategy
        case _:
            raise ValueError(f"Unsupported patching strategy: {patching_config.name}")

    # remove `name` to match the class signatures
    # tiling requires `tile_size` instead of `patch_size`, hence the aliasing
    return patch_class(
        data_shapes=data_shapes, **patching_config.model_dump(exclude={"name"})
    )
