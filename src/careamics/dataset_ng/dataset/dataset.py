from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import numpy as np
from config.support import SupportedData
from numpy._typing import NDArray
from patch_extractor import ImageStackLoader
from torch.utils.data import Dataset
from typing_extensions import ParamSpec

from careamics.config import DataConfig, InferenceConfig
from careamics.dataset.patching.patching import Stats
from careamics.dataset_ng.patch_extractor import (
    PatchExtractor,
    PatchSpecs,
    create_patch_extractor,
)
from careamics.dataset_ng.patching_strategies import (
    PatchSpecsGenerator,
    RandomPatchSpecsGenerator,
    TiledPatchSpecsGenerator,
)
from careamics.transforms import Compose

P = ParamSpec("P")


class ImageRegionData(NamedTuple):
    data: NDArray
    source: Path | Literal["array"]
    data_shape: Sequence[int]
    dtype: str  # dtype should be str for collate
    axes: str
    region_spec: PatchSpecs


InputType = Sequence[np.ndarray] | Sequence[Path]


class CareamicsDataset(Dataset):
    def __init__(
        self,
        data_config: DataConfig | InferenceConfig,
        inputs: InputType,
        targets: Optional[InputType] = None,
        image_stack_loader: Optional[ImageStackLoader[P]] = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self.config = data_config

        data_type_enum = SupportedData(self.config.data_type)
        self.input_extractor = create_patch_extractor(
            inputs,
            self.config.axes,
            data_type_enum,
            image_stack_loader,
            *args,
            **kwargs,
        )
        if targets is not None:
            self.target_extractor: Optional[PatchExtractor] = create_patch_extractor(
                targets,
                self.config.axes,
                data_type_enum,
                image_stack_loader,
                *args,
                **kwargs,
            )
        else:
            self.target_extractor = None

        self.patch_specs = self._initialize_patch_specs()

        self.input_stats, self.target_stats = self._initialize_statistics()

        self.transforms = self._initialize_transforms()

    def _initialize_patch_specs(self) -> list[PatchSpecs]:
        if isinstance(self.config, DataConfig):
            patch_generator: PatchSpecsGenerator = RandomPatchSpecsGenerator(
                patch_size=self.config.patch_size,
                random_seed=getattr(self.config, "random_seed", 42),
            )
        elif isinstance(self.config, InferenceConfig):
            # TODO: how to handle the whole images?
            if self.config.tile_size is None or self.config.tile_overlap is None:
                raise ValueError(
                    "InferenceConfig must specify tile_size and tile_overlap"
                )
            patch_generator = TiledPatchSpecsGenerator(
                patch_size=self.config.tile_size, overlap=self.config.tile_overlap
            )
        else:
            raise ValueError(f"Data config type {type(self.config)} is not supported.")

        patch_specs = patch_generator.generate(data_shapes=self.input_extractor.shape)
        return patch_specs

    def _initialize_transforms(self) -> Optional[Compose]:
        if isinstance(self.config, DataConfig):
            return Compose(
                transform_list=list(self.config.transforms),
            )
        # TODO: add TTA
        return None

    def _initialize_statistics(self) -> tuple[Stats, Stats | None]:
        # TODO: add running stats
        # Currently assume that stats are provided in the configuration
        input_stats = Stats(self.config.image_means, self.config.image_stds)
        target_stats = None
        if isinstance(self.config, DataConfig):
            target_means = self.config.target_means
            target_stds = self.config.target_stds
            if target_means is not None and target_stds is not None:
                target_stats = Stats(target_means, target_stds)
        return input_stats, target_stats

    def __len__(self):
        return len(self.patch_specs)

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        data_idx = patch_spec["data_idx"]
        return ImageRegionData(
            data=patch,
            source=extractor.image_stacks[data_idx].source,
            dtype=str(extractor.image_stacks[data_idx].data_dtype),
            data_shape=extractor.image_stacks[data_idx].data_shape,
            # TODO: should it be axes of the original image instead?
            axes=self.config.axes,
            region_spec=patch_spec,
        )

    def __getitem__(self, index: int) -> tuple[ImageRegionData, ImageRegionData | None]:
        patch_spec = self.patch_specs[index]
        input_patch = self.input_extractor.extract_patch(**patch_spec)

        target_patch = (
            self.target_extractor.extract_patch(**patch_spec)
            if self.target_extractor is not None
            else None
        )

        if self.transforms is not None:
            input_patch, target_patch = self.transforms(input_patch, target_patch)

        input_data = self._create_image_region(
            patch=input_patch, patch_spec=patch_spec, extractor=self.input_extractor
        )

        if target_patch is not None and self.target_extractor is not None:
            target_data = self._create_image_region(
                patch=target_patch,
                patch_spec=patch_spec,
                extractor=self.target_extractor,
            )
        else:
            target_data = None

        return input_data, target_data
