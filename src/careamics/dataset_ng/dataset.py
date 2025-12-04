from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config.data.ng_data_config import NGDataConfig, WholePatchingConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)
from careamics.transforms import Compose

from .image_stack import GenericImageStack
from .normalization import create_normalization
from .normalization.statistics import resolve_normalization_config
from .patch_extractor import PatchExtractor
from .patch_filter import create_coord_filter, create_patch_filter
from .patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    PatchSpecs,
    RandomPatchingStrategy,
    RegionSpecs,
    TilingStrategy,
    WholeSamplePatchingStrategy,
)


class Mode(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    PREDICTING = "predicting"


class ImageRegionData(NamedTuple, Generic[RegionSpecs]):
    data: NDArray
    source: Union[str, Literal["array"]]
    data_shape: Sequence[int]
    dtype: str  # dtype should be str for collate
    axes: str
    region_spec: RegionSpecs  # PatchSpecs or subclasses, e.g. TileSpecs

    chunks: Sequence[int] = (1,)  # default value for ImageStack without chunks


InputType = Union[Sequence[NDArray[Any]], Sequence[Path]]


def _patch_size_within_data_shapes(
    data_shapes: Sequence[Sequence[int]], patch_size: Sequence[int]
) -> bool:
    """Determine whether all the data_shapes are greater than the patch size.

    Parameters
    ----------
    data_shapes : Sequence[Sequence[int]]
        A sequence of data shapes. They must be in the format SC(Z)YX.
    patch_size : Sequence[int]
        A patch size that must specify the size of the patch in all the spatial
        dimensions, in the format (Z)YX.

    Returns
    -------
    bool
        If all the data shapes are greater than the patch size.
    """
    smaller_than_shapes = [
        # skip sample and channel dimension in data_shape
        (np.array(patch_size) < np.array(data_shape[2:])).all()
        for data_shape in data_shapes
    ]
    return all(smaller_than_shapes)


class CareamicsDataset(Dataset, Generic[GenericImageStack]):
    def __init__(
        self,
        data_config: NGDataConfig,
        mode: Mode,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: PatchExtractor[GenericImageStack] | None = None,
        mask_extractor: PatchExtractor[GenericImageStack] | None = None,
    ) -> None:

        # Make sure all the image sizes are greater than the patch size for training
        data_shapes = [
            image_stack.data_shape for image_stack in input_extractor.image_stacks
        ]
        if mode != Mode.PREDICTING:
            if not isinstance(
                data_config.patching, WholePatchingConfig
            ) and not _patch_size_within_data_shapes(
                data_shapes, data_config.patching.patch_size
            ):
                raise ValueError(
                    "Not all images sizes are greater than the patch size for training "
                    "and validation."
                )

        self.config = data_config
        self.mode = mode

        self.input_extractor = input_extractor
        self.target_extractor = target_extractor

        self.patch_filter = (
            create_patch_filter(self.config.patch_filter)
            if self.config.patch_filter is not None
            else None
        )
        self.coord_filter = (
            create_coord_filter(self.config.coord_filter, mask=mask_extractor)
            if self.config.coord_filter is not None and mask_extractor is not None
            else None
        )
        self.patch_filter_patience = self.config.patch_filter_patience

        self.patching_strategy = self._initialize_patching_strategy()

        self._resolved_norm_config = resolve_normalization_config(
            norm_config=self.config.normalization,
            patching_strategy=self.patching_strategy,
            input_extractor=self.input_extractor,
            target_extractor=self.target_extractor,
        )
        self.normalization = create_normalization(self._resolved_norm_config)
        
        self.transforms = self._initialize_transforms()

    def _initialize_patching_strategy(self) -> PatchingStrategy:
        patching_strategy: PatchingStrategy
        if self.mode == Mode.TRAINING:
            if self.config.patching.name != SupportedPatchingStrategy.RANDOM:
                raise ValueError(
                    f"Only `random` patching strategy supported during training, got "
                    f"{self.config.patching.name}."
                )

            patching_strategy = RandomPatchingStrategy(
                data_shapes=self.input_extractor.shape,
                patch_size=self.config.patching.patch_size,
                seed=self.config.seed,
            )
        elif self.mode == Mode.VALIDATING:
            if self.config.patching.name != SupportedPatchingStrategy.RANDOM:
                raise ValueError(
                    f"Only `random` patching strategy supported during training, got "
                    f"{self.config.patching.name}."
                )

            patching_strategy = FixedRandomPatchingStrategy(
                data_shapes=self.input_extractor.shape,
                patch_size=self.config.patching.patch_size,
                seed=self.config.seed,
            )
        elif self.mode == Mode.PREDICTING:
            if (
                self.config.patching.name != SupportedPatchingStrategy.TILED
                and self.config.patching.name != SupportedPatchingStrategy.WHOLE
            ):
                raise ValueError(
                    f"Only `tiled` and `whole` patching strategy supported during "
                    f"training, got {self.config.patching.name}."
                )
            elif self.config.patching.name == SupportedPatchingStrategy.TILED:
                patching_strategy = TilingStrategy(
                    data_shapes=self.input_extractor.shape,
                    tile_size=self.config.patching.patch_size,
                    overlaps=self.config.patching.overlaps,
                )
            else:
                patching_strategy = WholeSamplePatchingStrategy(
                    data_shapes=self.input_extractor.shape
                )
        else:
            raise ValueError(f"Unrecognised dataset mode {self.mode}.")

        return patching_strategy

    def _initialize_transforms(self) -> Compose | None:
        if self.mode == Mode.TRAINING:
            return Compose(list(self.config.transforms))

        # TODO: add TTA
        return None
    
    def __len__(self):
        return self.patching_strategy.n_patches

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        data_idx = patch_spec["data_idx"]
        image_stack: GenericImageStack = extractor.image_stacks[data_idx]
        return ImageRegionData(
            data=patch,
            source=str(image_stack.source),
            dtype=str(image_stack.data_dtype),
            data_shape=image_stack.data_shape,
            chunks=image_stack.chunks,
            # TODO: should it be axes of the original image instead?
            axes=self.config.axes,
            region_spec=patch_spec,
        )

    def _extract_patches(
        self, patch_spec: PatchSpecs
    ) -> tuple[NDArray, NDArray | None]:
        """Extract input and target patches based on patch specifications."""
        input_patch = self.input_extractor.extract_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )

        target_patch = (
            self.target_extractor.extract_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            if self.target_extractor is not None
            else None
        )
        return input_patch, target_patch

    def _get_filtered_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, PatchSpecs]:
        """Extract a patch that passes filtering criteria with retry logic."""
        should_filter = self.mode == Mode.TRAINING and (
            self.patch_filter is not None or self.coord_filter is not None
        )
        empty_patch = True
        patch_filter_patience = self.patch_filter_patience  # reset patience

        while empty_patch and patch_filter_patience > 0:
            # query patches
            patch_spec = self.patching_strategy.get_patch_spec(index)

            # filter patch based on coordinates if needed
            if should_filter and self.coord_filter is not None:
                if self.coord_filter.filter_out(patch_spec):
                    patch_filter_patience -= 1

                    # TODO should we raise an error rather than silently accept patches?
                    # if patience runs out without ever finding coordinates
                    # then we need to guard against an exist before defining
                    # input_patch and target_patch
                    if patch_filter_patience != 0:
                        continue

            input_patch, target_patch = self._extract_patches(patch_spec)

            # filter patch based on values if needed
            if should_filter and self.patch_filter is not None:
                empty_patch = self.patch_filter.filter_out(input_patch)
                patch_filter_patience -= 1  # decrease patience
            else:
                empty_patch = False

        return input_patch, target_patch, patch_spec

    def __getitem__(
        self, index: int
    ) -> Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]]:
        input_patch, target_patch, patch_spec = self._get_filtered_patch(index)

        # apply normalization
        input_patch, target_patch = self.normalization(input_patch, target_patch)

        # apply transforms
        if self.transforms is not None:
            if self.target_extractor is not None:
                input_patch, target_patch = self.transforms(input_patch, target_patch)
            else:
                # TODO: compose doesn't return None for target patch anymore
                #   so have to do this annoying if else
                (input_patch,) = self.transforms(input_patch, target_patch)
                target_patch = None

        input_data = self._create_image_region(
            patch=input_patch, patch_spec=patch_spec, extractor=self.input_extractor
        )

        if target_patch is not None and self.target_extractor is not None:
            target_data = self._create_image_region(
                patch=target_patch,
                patch_spec=patch_spec,
                extractor=self.target_extractor,
            )
            return input_data, target_data
        else:
            return (input_data,)
