from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing_extensions import ParamSpec

from careamics.config import DataConfig, InferenceConfig
from careamics.config.support import SupportedData
from careamics.dataset.patching.patching import Stats
from careamics.dataset_ng.patch_extractor import (
    ImageStackLoader,
    PatchExtractor,
    create_patch_extractor,
)
from careamics.dataset_ng.patching_strategies import (
    FixedRandomPatchingStrategy,
    PatchingStrategy,
    PatchSpecs,
    RandomPatchingStrategy,
)
from careamics.transforms import Compose

P = ParamSpec("P")


class Mode(str, Enum):
    TRAINING = "training"
    VALIDATING = "validating"
    PREDICTING = "predicting"


class ImageRegionData(NamedTuple):
    data: NDArray
    source: Union[Path, Literal["array"]]
    data_shape: Sequence[int]
    dtype: str  # dtype should be str for collate
    axes: str
    region_spec: PatchSpecs


InputType = Union[Sequence[np.ndarray], Sequence[Path]]


class CareamicsDataset(Dataset):
    def __init__(
        self,
        data_config: Union[DataConfig, InferenceConfig],
        mode: Mode,
        inputs: InputType,
        targets: Optional[InputType] = None,
        image_stack_loader: Optional[ImageStackLoader[P]] = None,
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        self.config = data_config
        self.mode = mode

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
            # TODO: patching strategy will be tilingStrategy in upcoming PR
            raise NotImplementedError(
                "Prediction mode for the CAREamicsDataset has not been implemented yet."
            )
        else:
            raise ValueError(f"Unrecognised dataset mode {self.mode}.")

        return patching_strategy

    def _initialize_transforms(self) -> Optional[Compose]:
        if isinstance(self.config, DataConfig):
            return Compose(
                transform_list=list(self.config.transforms),
            )
        # TODO: add TTA
        return None

    def _initialize_statistics(self) -> tuple[Stats, Optional[Stats]]:
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
        return self.patching_strategy.n_patches

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

    def __getitem__(
        self, index: int
    ) -> tuple[ImageRegionData, Optional[ImageRegionData]]:
        patch_spec = self.patching_strategy.get_patch_spec(index)
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
