from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

from careamics.dataset.image_stack import GenericImageStack
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import Patching, PatchSpecs

from .metadata_utils import ImageMetadata, get_image_metadata
from .patch_constructor import PatchConstructor


class BasicPatchConstructor(PatchConstructor):
    def __init__(
        self,
        patching_strategy: Patching,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: PatchExtractor[GenericImageStack] | None = None,
        channels: Sequence[int] | None = None,
    ):
        self.input_extractor = input_extractor
        self.target_extractor = target_extractor
        self.patching_strategy = patching_strategy
        self.channels = channels

    @property
    def n_patches(self):
        return self.patching_strategy.n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        return self.input_extractor.shapes

    @property
    def target_shapes(self) -> Sequence[Sequence[int]] | None:
        if self.target_extractor is not None:
            return self.target_extractor.shapes
        return None

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, PatchSpecs]:
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch, target_patch = self._extract_patches(patch_spec)
        return input_patch, target_patch, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        return input_patch

    def get_input_image_metadata(self, data_idx: int) -> ImageMetadata:
        image_stack = self.input_extractor.image_stacks[data_idx]
        return get_image_metadata(image_stack)

    def get_target_image_metadata(self, data_idx: int) -> ImageMetadata | None:
        if self.target_extractor is not None:
            image_stack = self.target_extractor.image_stacks[data_idx]
            return get_image_metadata(image_stack)
        return None

    def _extract_patches(
        self, patch_spec: PatchSpecs
    ) -> tuple[NDArray[Any], NDArray[Any] | None]:
        """Extract input and target patches based on patch specifications.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification (data_idx, sample_idx, coords, patch_size).

        Returns
        -------
        tuple of (NDArray, NDArray or None)
            Input patch and optional target patch.
        """
        input_patch = self.input_extractor.extract_channel_patch(
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            channels=self.channels,
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
        )

        target_patch = (
            self.target_extractor.extract_channel_patch(
                data_idx=patch_spec["data_idx"],
                sample_idx=patch_spec["sample_idx"],
                # TODO does not allow selecting different channels for target
                channels=self.channels,
                coords=patch_spec["coords"],
                patch_size=patch_spec["patch_size"],
            )
            if self.target_extractor is not None
            else None
        )
        return input_patch, target_patch
