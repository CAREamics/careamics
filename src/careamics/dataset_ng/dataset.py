from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, Optional, Union
from typing import Generic, Literal, NamedTuple, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.config.transformations import NormalizeModel
from careamics.dataset.patching.patching import Stats
from careamics.dataset_ng.patch_extractor import PatchExtractor
from careamics.dataset_ng.patch_extractor.image_stack import GenericImageStack
from careamics.dataset_ng.patch_extractor import (
    ImageStackLoader,
    PatchExtractor,
    create_patch_extractor,
)
from careamics.dataset_ng.patch_extractor.image_stack import ImageStack
from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    PatchSpecs,
    RandomPatchingStrategy,
    TileSpecs,
    TilingStrategy,
    WholeSamplePatchingStrategy,
)
from careamics.transforms import Compose


class Mode(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    PREDICTING = "predicting"


class ImageRegionData(NamedTuple):
    data: NDArray
    source: Union[str, Literal["array"]]
    data_shape: Sequence[int]
    dtype: str  # dtype should be str for collate
    axes: str
    region_spec: PatchSpecs


InputType = Union[Sequence[NDArray[Any]], Sequence[Path]]


class CareamicsDataset(Dataset, Generic[GenericImageStack]):
    def __init__(
        self,
        data_config: Union[DataConfig, InferenceConfig],
        mode: Mode,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: Optional[PatchExtractor[GenericImageStack]] = None,
    ):
        self.config = data_config
        self.mode = mode

        self.input_extractor = input_extractor
        self.target_extractor = target_extractor

        self.patching_strategy = self._initialize_patching_strategy()

        self.input_stats, self.target_stats = self._initialize_statistics()

        self.transforms = self._initialize_transforms()

    def _initialize_patching_strategy(self) -> PatchingStrategy:
        patching_strategy: PatchingStrategy
        if self.mode == Mode.TRAINING:
            if isinstance(self.config, InferenceConfig):
                raise ValueError("Inference config cannot be used for training.")
            patching_strategy = RandomPatchingStrategy(
                data_shapes=self.input_extractor.shape,
                patch_size=self.config.patch_size,
                # TODO: Add random seed to dataconfig
                seed=getattr(self.config, "random_seed", None),
            )
        elif self.mode == Mode.VALIDATING:
            if isinstance(self.config, InferenceConfig):
                raise ValueError("Inference config cannot be used for validating.")
            patching_strategy = FixedRandomPatchingStrategy(
                data_shapes=self.input_extractor.shape,
                patch_size=self.config.patch_size,
                # TODO: Add random seed to dataconfig
                seed=getattr(self.config, "random_seed", None),
            )
        elif self.mode == Mode.PREDICTING:
            if not isinstance(self.config, InferenceConfig):
                raise ValueError("Inference config must be used for predicting.")
            if (self.config.tile_size is not None) and (
                self.config.tile_overlap is not None
            ):
                patching_strategy = TilingStrategy(
                    data_shapes=self.input_extractor.shape,
                    tile_size=self.config.tile_size,
                    overlaps=self.config.tile_overlap,
                )
            else:
                patching_strategy = WholeSamplePatchingStrategy(
                    data_shapes=self.input_extractor.shape
                )
        else:
            raise ValueError(f"Unrecognised dataset mode {self.mode}.")

        return patching_strategy

    def _initialize_transforms(self) -> Optional[Compose]:
        if isinstance(self.config, DataConfig):
            if self.mode == Mode.TRAINING:
                # TODO: initialize normalization separately depending on configuration
                return Compose(
                    transform_list=[
                        NormalizeModel(
                            image_means=self.input_stats.means,
                            image_stds=self.input_stats.stds,
                            target_means=self.target_stats.means,
                            target_stds=self.target_stats.stds,
                        )
                    ]
                    + list(self.config.transforms)
                )

            else:
                return Compose(
                    transform_list=[
                        NormalizeModel(
                            image_means=self.input_stats.means,
                            image_stds=self.input_stats.stds,
                            target_means=self.target_stats.means,
                            target_stds=self.target_stats.stds,
                        )
                    ]
                )

        # TODO: add TTA
        return None

    def _initialize_statistics(self) -> tuple[Stats, Optional[Stats]]:
        # TODO: add running stats
        # Currently assume that stats are provided in the configuration
        input_stats = Stats(self.config.image_means, self.config.image_stds)

        if type(self.config) == DataConfig:
            target_means = self.config.target_means
            target_stds = self.config.target_stds
        else:
            target_means = None
            target_stds = None
        if target_means is not None and target_stds is not None:
            target_stats = Stats(target_means, target_stds)
        else:
            target_stats = Stats((), ())

        return input_stats, target_stats

    def __len__(self):
        return self.patching_strategy.n_patches

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        data_idx = patch_spec["data_idx"]
        source = extractor.image_stacks[data_idx].source
        return ImageRegionData(
            data=patch,
            source=str(source),
            dtype=str(extractor.image_stacks[data_idx].data_dtype),
            data_shape=extractor.image_stacks[data_idx].data_shape,
            # TODO: should it be axes of the original image instead?
            axes=self.config.axes,
            region_spec=patch_spec,
        )

    def __getitem__(
        self, index: int
    ) -> tuple[ImageRegionData, Optional[ImageRegionData]]:
        patch_spec = self.patching_strategy.get_patch_spec(index)
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

        return input_data
