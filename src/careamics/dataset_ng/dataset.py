from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from careamics.config.data.ng_data_model import NGDataConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)
from careamics.config.transformations import NormalizeModel
from careamics.dataset.dataset_utils.running_stats import WelfordStatistics
from careamics.dataset.patching.patching import Stats
from careamics.dataset_ng.patch_extractor import GenericImageStack, PatchExtractor
from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    PatchSpecs,
    RandomPatchingStrategy,
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
        data_config: NGDataConfig,
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

    def _initialize_transforms(self) -> Optional[Compose]:
        normalize = NormalizeModel(
            image_means=self.input_stats.means,
            image_stds=self.input_stats.stds,
            target_means=self.target_stats.means,
            target_stds=self.target_stats.stds,
        )
        if self.mode == Mode.TRAINING:
            # TODO: initialize normalization separately depending on configuration
            return Compose(transform_list=[normalize] + list(self.config.transforms))

        # TODO: add TTA
        return Compose(transform_list=[normalize])

    def _calculate_stats(
        self, data_extractor: PatchExtractor[GenericImageStack]
    ) -> Stats:
        image_stats = WelfordStatistics()
        n_patches = self.patching_strategy.n_patches

        for idx in tqdm(range(n_patches), desc="Computing statistics"):
            patch_spec = self.patching_strategy.get_patch_spec(idx)
            patch = data_extractor.extract_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            # TODO: statistics accept SCYX format, while patch is CYX
            image_stats.update(patch[None, ...], sample_idx=idx)

        image_means, image_stds = image_stats.finalize()
        return Stats(image_means, image_stds)

    # TODO: add running stats
    def _initialize_statistics(self) -> tuple[Stats, Stats]:
        if self.config.image_means is not None and self.config.image_stds is not None:
            input_stats = Stats(self.config.image_means, self.config.image_stds)
        else:
            input_stats = self._calculate_stats(self.input_extractor)

        target_stats = Stats((), ())

        if self.config.target_means is not None and self.config.target_stds is not None:
            target_stats = Stats(self.config.target_means, self.config.target_stds)
        elif self.target_extractor is not None:
            target_stats = self._calculate_stats(self.target_extractor)

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
    ) -> Union[tuple[ImageRegionData], tuple[ImageRegionData, ImageRegionData]]:
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
        else:
            return (input_data,)
