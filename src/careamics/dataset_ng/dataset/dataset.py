from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional

import numpy as np
from numpy._typing import NDArray
from torch.utils.data import Dataset

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
    SequentialPatchSpecsGenerator,
)
from careamics.transforms import Compose


class ImageRegionData(NamedTuple):
    data: NDArray
    filename: Path | Literal["array"]
    data_shape: Sequence[int]
    dtype: str  # dtype should be str for collate
    axes: str
    # TODO: what to do in case of inference on the whole image?
    #  patch spec of whole image or none?
    region_spec: PatchSpecs


# TODO: how to handle custom data types?
InputType = Sequence[np.ndarray] | Sequence[Path]


class CareamicsDataset(Dataset):
    # TODO: dataset train / val splitting, should it be in an upper layer?
    # TODO: if validation is initialized separately, how to disable transforms?
    # TODO: create file description
    def __init__(
        self,
        data_config: DataConfig | InferenceConfig,
        inputs: Optional[InputType] = None,
        targets: Optional[InputType] = None,
        read_func: Optional[Callable] = None,
    ):
        self.config = data_config
        if inputs is None and targets is None:
            raise ValueError(
                "For splitting tasks either real input or target channels are required."
                "For all other tasks input is required."
            )
        # TODO check if inputs and targets are path or sequence of arrays, not a single array
        if inputs is not None:
            self.input_extractor: Optional[PatchExtractor] = create_patch_extractor(
                self.config, data=inputs, read_func=read_func
            )
            self.data_shapes = [ie.shape for ie in self.input_extractors]

        if targets is not None:
            self.target_extractor: Optional[PatchExtractor] = create_patch_extractor(
                self.config, data=targets, read_func=read_func
            )
            self.data_shapes = (
                self.target_extractor.shape
                if not hasattr(self, "input_extractor")
                else self.data_shapes
            )

        # TODO check that the list of coords is different for all epochs
        self.patch_specs = self._initialize_patch_specs()

        self.input_stats, self.target_stats = self._initialize_statistics()

        self.transforms = self._initialize_transforms()

    def _initialize_patch_specs(self) -> list[PatchSpecs]:
        if isinstance(self.config, DataConfig):
            # TODO: how to fix the random seed properly?
            #  how to change it between epochs?
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
            patch_generator = SequentialPatchSpecsGenerator(
                patch_size=self.config.tile_size, overlap=self.config.tile_overlap
            )
        else:
            raise ValueError(f"Data config type {type(self.config)} is not supported.")

        patch_specs = patch_generator.generate(
            data_shapes=self.data_shapes,
            collocate_patch_region=self.config.patch_extractor_params.collocate_patch_region,
        )
        return patch_specs

    def _initialize_transforms(self) -> Optional[Compose]:
        if self.config.transforms:
            return Compose(
                transform_list=list(self.config.transforms),
            )
        # TODO: add TTA
        return None

    def _initialize_statistics(self) -> tuple[Stats, Stats | None]:
        # TODO: add running stats
        # Currently assume that stats are provided in the configuration
        input_stats = Stats(self.config.image_means, self.config.image_stds)
        target_means = getattr(self.config, "target_means", None)
        target_stds = getattr(self.config, "target_stds", None)
        if target_means is not None and target_stds is not None:
            target_stats = Stats(target_means, target_stds)
        else:
            target_stats = None
        return input_stats, target_stats

    def __len__(self):
        return len(self.patch_specs)

    def _create_image_region(
        self, patch: np.ndarray, patch_spec: PatchSpecs, extractor: PatchExtractor
    ) -> ImageRegionData:
        data_idx = patch_spec["data_idx"]
        return ImageRegionData(
            data=patch,
            filename=extractor.image_stacks[data_idx].source,
            dtype=str(extractor.image_stacks[data_idx].data_dtype),
            data_shape=extractor.image_stacks[data_idx].data_shape,
            # TODO: should it be axes of the original image instead?
            axes=self.config.axes,
            region_spec=patch_spec,
        )

    def __getitem__(self, index: int) -> tuple[ImageRegionData, ImageRegionData | None]:
        patch_spec = self.patch_specs[index]
        # TODO add 'meta' patch spec to keep track of ps for all channels, or for loop in init
        input_patch = (
            self.input_extractor.extract_patch(**patch_spec)
            if hasattr(self, "input_extractor")
            else None
        )

        target_patch = (
            self.target_extractor.extract_patch(**patch_spec)
            if hasattr(self, "target_extractor")
            else None
        )

        # TODO patch mixing, add mixer as a transform or just a mixer?

        if self.transforms is not None:
            input_patch, target_patch = self.transforms(input_patch, target_patch)

        input_data = (
            self._create_image_region(
                patch=input_patch, patch_spec=patch_spec, extractor=self.input_extractor
            )
            if input_patch is not None
            else None
        )

        target_data = (
            self._create_image_region(
                patch=target_patch,
                patch_spec=patch_spec,
                extractor=self.target_extractor,
            )
            if target_patch is not None
            else None
        )
        # TODO nones won't stack ?

        return input_data, target_data
