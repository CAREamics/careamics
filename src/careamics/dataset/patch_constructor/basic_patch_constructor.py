"""Basic patch constructor implementation."""

from collections.abc import Sequence
from typing import Any

from numpy.typing import NDArray

from careamics.dataset.image_stack import GenericImageStack
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import Patching, PatchSpecs

from .metadata_utils import ImageMetadata, get_image_metadata
from .patch_constructor import PatchConstructor


class BasicPatchConstructor(PatchConstructor):
    """Construct standard CAREamics input and optional target patches.

    This constructor can be used for both training and prediction pipelines.

    Parameters
    ----------
    patching_strategy : Patching
        Strategy that maps dataset indices to patch specifications.
    input_extractor : PatchExtractor
        Extractor for input patches.
    target_extractor : PatchExtractor or None, default=None
        Extractor for target patches. Can be `None` for self-supervised algorithms or
        prediction pipelines.
    channels : Sequence[int] or None, default=None
        Channel indices to extract. If `None`, all channels are extracted.
    """

    def __init__(
        self,
        patching_strategy: Patching,
        input_extractor: PatchExtractor[GenericImageStack],
        target_extractor: PatchExtractor[GenericImageStack] | None = None,
        channels: Sequence[int] | None = None,
    ):
        """Initialize the basic patch constructor.

        Parameters
        ----------
        patching_strategy : Patching
            Strategy that maps dataset indices to patch specifications.
        input_extractor : PatchExtractor
            Extractor for input patches.
        target_extractor : PatchExtractor or None, default=None
            Extractor for target patches. Can be `None` for self-supervised algorithms
            or prediction pipelines.
        channels : Sequence[int] or None, default=None
            Channel indices to extract. If `None`, all channels are extracted.
        """
        self.input_extractor = input_extractor
        self.target_extractor = target_extractor
        self.patching_strategy = patching_strategy
        self.channels = channels

    @property
    def n_patches(self):
        """Return the number of available patches.

        Returns
        -------
        int
            Number of available patches.
        """
        return self.patching_strategy.n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """Return input image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Input image shapes, SC(Z)YX format.
        """
        return self.input_extractor.shapes

    @property
    def target_shapes(self) -> Sequence[Sequence[int]] | None:
        """Return target image shapes, if targets exist.

        Returns
        -------
        Sequence[Sequence[int]] or None
            Target image shapes in SC(Z)YX format, if targets exist, otherwise `None`.
        """
        if self.target_extractor is not None:
            return self.target_extractor.shapes
        return None

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, PatchSpecs]:
        """Construct the input and optional target patch for an index.

        Parameters
        ----------
        index : int
            Dataset index to map to a patch specification.

        Returns
        -------
        input_patch : NDArray[Any]
            Input patch with axes C(Z)YX.
        target_patch : NDArray[Any] or None
            Target patch with axes C(Z)YX when a target extractor is available,
            otherwise `None`.
        patch_spec : PatchSpecs
            Patch specification used to extract the patches.
        """
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch, target_patch = self._extract_patches(patch_spec)
        return input_patch, target_patch, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Return the principal input patch.

        The principal input is used for calculating statistics for normalization.

        Parameters
        ----------
        input_patch : NDArray[Any]
            Input patch with axes C(Z)YX.

        Returns
        -------
        NDArray[Any]
            The unchanged input patch with axes C(Z)YX.
        """
        return input_patch

    def get_input_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata:
        """Return metadata for the input image described by a patch specification.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification containing the input image index.

        Returns
        -------
        ImageMetadata
            Metadata for the input image.
        """
        data_idx = patch_spec["data_idx"]
        image_stack = self.input_extractor.image_stacks[data_idx]
        return get_image_metadata(image_stack)

    def get_target_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata | None:
        """Return metadata for the target image described by a patch specification.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification containing the target image index.

        Returns
        -------
        ImageMetadata | None
            Metadata for the target image when targets exist, otherwise `None`.
        """
        data_idx = patch_spec["data_idx"]
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
        input_patch : NDArray[Any]
            Input patch with axes C(Z)YX.
        target_patch : NDArray[Any] or None
            Target patch with axes C(Z)YX when a target extractor is available,
            otherwise `None`.
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
