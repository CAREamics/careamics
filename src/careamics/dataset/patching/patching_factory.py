"""Patching strategy factory."""

from collections.abc import Sequence

from careamics.config.data.data_config import PatchingConfig
from careamics.config.data.patching_strategies import (
    FixedRandomPatchingConfig,
    RandomPatchingConfig,
    StratifiedPatchingConfig,
    TiledPatchingConfig,
    WholePatchingConfig,
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
    parameters = patching_config.model_dump(exclude={"name"})
    # from PEP 634, Class patterns
    # if no arguments are present, the pattern succeeds if
    # the isinstance() check succeeds.
    match patching_config:
        case RandomPatchingConfig():
            return RandomPatching(
                data_shapes=data_shapes,
                **parameters,
            )
        case StratifiedPatchingConfig():
            return StratifiedPatching(
                data_shapes=data_shapes,
                **parameters,
            )
        case FixedRandomPatchingConfig():
            return FixedRandomPatching(
                data_shapes=data_shapes,
                **parameters,
            )
        case TiledPatchingConfig():
            return TiledPatching(
                data_shapes=data_shapes,
                **parameters,
            )
        case WholePatchingConfig():
            return WholeSamplePatching(
                data_shapes=data_shapes,
                **parameters,
            )
        case _:
            raise ValueError(f"Unsupported patching: {patching_config.name}")
