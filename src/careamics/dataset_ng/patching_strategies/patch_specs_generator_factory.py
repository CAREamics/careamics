from collections.abc import Sequence
from typing import Union

from careamics.config import DataConfig, InferenceConfig

from .patch_specs_generator import (
    PatchSpecsGenerator,
    RandomPatchSpecsGenerator,
    SequentialPatchSpecsGenerator,
)


def create_patch_specs_generator(
    data_config: Union[DataConfig, InferenceConfig],
    data_shapes: Sequence[Sequence[int]],
    **kwargs,
) -> PatchSpecsGenerator:
    """Create a patch specs generator based on configuration.

    Args:
        data_config: Configuration object containing patching parameters
        data_shapes: Shapes of the input data
        **kwargs: Additional arguments passed to specific generators
    """
    if isinstance(data_config, DataConfig):
        # TODO: how to fix the random seed properly? how to change it between epochs?
        return RandomPatchSpecsGenerator(
            data_shapes=data_shapes, random_seed=getattr(data_config, "random_seed", 42)
        )
    elif isinstance(data_config, InferenceConfig):
        if not hasattr(data_config, "tile_overlap"):
            raise ValueError("InferenceConfig must specify tile_overlap")
        return SequentialPatchSpecsGenerator(
            data_shapes=data_shapes, overlap=data_config.tile_overlap
        )
    else:
        raise ValueError(f"Data config type {type(data_config)} is not supported.")
