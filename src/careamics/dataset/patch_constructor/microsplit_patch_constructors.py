"""Patch constructors for MicroSplit dataset modes."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from careamics.dataset.image_stack import ImageStack
from careamics.dataset.patch_extractor import PatchExtractor
from careamics.dataset.patching import (
    Patching,
    PatchSpecs,
    UncorrelatedPatchSpecs,
    is_uncorrelated_specs,
)

from .metadata_utils import ImageMetadata, get_image_metadata
from .patch_constructor import PatchConstructor

# placeholder names (correspond to Training Modes I, II & III described in the paper)


# target channels acquired together, synthetic sum input
class MsT1PatchConstructor(PatchConstructor):
    """Construct MicroSplit patches from multiplexed images of target channels.

    Synthetic inputs created by summing the target channels, `alpha_ranges` are used
    to decide the weight of each channel.

    Parameters
    ----------
    patching_strategy : Patching
        Strategy that maps dataset indices to patch specifications.
    target_extractor : PatchExtractor
        Extractor for multiplexed target channels.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}
        Padding mode used when lateral context extends beyond image boundaries.
    alpha_ranges : Sequence[tuple[float, float]] or None
        Sampling ranges for channel mixing weights. If `None`, every channel gets a
        fixed weight of `1 / n_channels`.
    uncorrelated_channel_prob : float
        Probability of sampling channels from different patch locations.
    channels : Sequence[int] or None, default=None
        Channel indices to use from the target data. If `None`, all channels are used.
    rng : numpy.random.Generator or None, default=None
        Random number generator. If `None`, a default generator is created.
    """

    def __init__(
        self,
        patching_strategy: Patching,
        target_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
        alpha_ranges: Sequence[tuple[float, float]] | None,
        uncorrelated_channel_prob: float,
        channels: Sequence[int] | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the training mode I patch constructor.

        Parameters
        ----------
        patching_strategy : Patching
            Strategy that maps dataset indices to patch specifications.
        target_extractor : PatchExtractor
            Extractor for jointly acquired target channels.
        multiscale_count : int
            Number of lateral context inputs.
        padding_mode : {"reflect", "wrap"}
            Padding mode used when lateral context extends beyond image boundaries.
        alpha_ranges : Sequence[tuple[float, float]] or None
            Sampling ranges for channel mixing weights. If `None`, every channel gets
            a fixed weight of `1 / n_channels`.
        uncorrelated_channel_prob : float
            Probability of sampling channels from different patch locations.
        channels : Sequence[int] or None, default=None
            Channel indices to use from the target data. If `None`, all channels are
            used.
        rng : numpy.random.Generator or None, default=None
            Random number generator. If `None`, a default generator is created.
        """
        self.rng = rng if rng is not None else np.random.default_rng()

        self.patching_strategy = patching_strategy
        self.target_extractor = target_extractor
        self.channels = channels
        self.alpha_ranges = alpha_ranges
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode
        self.uncorrelated_channel_prob = uncorrelated_channel_prob

    @property
    def n_patches(self) -> int:
        """Return the number of available patches.

        Returns
        -------
        int
            Number of patches available from the patching strategy.
        """
        return self.patching_strategy.n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """Return input image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Input image shapes.
        """
        return self.target_extractor.shapes

    @property
    def target_shapes(self) -> Sequence[Sequence[int]]:
        """Return target image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Target image shapes.
        """
        return self.target_extractor.shapes

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], PatchSpecs | UncorrelatedPatchSpecs]:
        """Construct the synthetic input and target patch for an index.

        Parameters
        ----------
        index : int
            Dataset index to map to a patch specification.

        Returns
        -------
        input_patch : NDArray[Any]
            Synthetic input patch with axes L(Z)YX, where L is the lateral context
            input axis ordered from the native patch scale to larger context scales.
        target_patch : NDArray[Any]
            Target patch with axes C(Z)YX.
        patch_spec : PatchSpecs or UncorrelatedPatchSpecs
            Standard patch specification for correlated channels, or uncorrelated
            patch specification when channels are sampled from different locations.
        """
        p = self.rng.random()
        patch_spec: PatchSpecs | UncorrelatedPatchSpecs
        if p < self.uncorrelated_channel_prob:
            n_channels = (
                self.target_extractor.n_channels
                if self.channels is None
                else len(self.channels)
            )
            # TODO: randomly sampling indices will not work well with iterable files
            indices: list[int] = [index] + self.rng.integers(
                0, self.n_patches, n_channels - 1
            ).tolist()
            patch_specs = [self.patching_strategy.get_patch_spec(i) for i in indices]
            patch, patch_spec = _construct_uncorrelated_patch(
                self.target_extractor,
                patch_specs,
                channels=self.channels,
                principal_channel=0,
                multiscale_count=self.multiscale_count,
                padding_mode=self.padding_mode,
            )
        else:
            patch_spec = self.patching_strategy.get_patch_spec(index)
            patch = _extract_lc_patch(
                self.target_extractor,
                **patch_spec,
                channels=self.channels,
                multiscale_count=self.multiscale_count,
                padding_mode=self.padding_mode,
            )
        # patch has axes CL(Z)YX

        input_patch, target_patch = _create_input_target(
            patch, self.alpha_ranges, self.rng
        )
        return input_patch, target_patch, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Return the principal input without lateral context.

        Parameters
        ----------
        input_patch : NDArray[Any]
            Input patch with axes L(Z)YX, where L is the lateral context input axis.

        Returns
        -------
        NDArray[Any]
            Principal input with axes C(Z)YX.
        """
        return input_patch[[0]]

    def get_input_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata:
        """Return metadata for the input image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification identifying the input image. Uncorrelated patch specs
            include metadata for each channel location.

        Returns
        -------
        ImageMetadata
            Metadata for the input image or principal uncorrelated input image.
        """
        if is_uncorrelated_specs(patch_spec):
            metadata = _get_uncorrelated_metadata(self.target_extractor, patch_spec)
            return metadata
        else:
            data_idx = patch_spec["data_idx"]
            image_stack = self.target_extractor.image_stacks[data_idx]
            return get_image_metadata(image_stack)

    def get_target_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata | None:
        """Return metadata for the target image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification identifying the target image. Uncorrelated patch specs
            include metadata for each channel location.

        Returns
        -------
        ImageMetadata
            Metadata for the target image or principal uncorrelated target image.
        """
        if is_uncorrelated_specs(patch_spec):
            metadata = _get_uncorrelated_metadata(self.target_extractor, patch_spec)
            return metadata
        else:
            data_idx = patch_spec["data_idx"]
            image_stack = self.target_extractor.image_stacks[data_idx]
            return get_image_metadata(image_stack)


# target channels in separate files, synthetic input
class MsT2PatchConstructor(PatchConstructor):
    """Construct MicroSplit patches from target channels in separate files.

    The data for different target channels may have different shapes.

    Synthetic inputs are created by summing patches from random locations,
    `alpha_ranges` are used to decide the weight of each channel.

    Parameters
    ----------
    patching_strategies : Sequence[Patching]
        One patching strategy per target channel source.
    target_extractors : Sequence[PatchExtractor]
        One extractor per target channel source.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}
        Padding mode used when lateral context extends beyond image boundaries.
    alpha_ranges : Sequence[tuple[float, float]] or None
        Sampling ranges for channel mixing weights. If `None`, every target source
        gets a fixed weight of `1 / n_channels`.
    rng : numpy.random.Generator or None, default=None
        Random number generator. If `None`, a default generator is created.
    """

    def __init__(
        self,
        patching_strategies: Sequence[Patching],
        target_extractors: Sequence[PatchExtractor[Any]],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
        alpha_ranges: Sequence[tuple[float, float]] | None,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the training mode II patch constructor.

        Parameters
        ----------
        patching_strategies : Sequence[Patching]
            One patching strategy per target channel source.
        target_extractors : Sequence[PatchExtractor]
            One extractor per target channel source.
        multiscale_count : int
            Number of lateral context inputs.
        padding_mode : {"reflect", "wrap"}
            Padding mode used when lateral context extends beyond image boundaries.
        alpha_ranges : Sequence[tuple[float, float]] or None
            Sampling ranges for channel mixing weights. If `None`, every target source
            gets a fixed weight of `1 / n_channels`.
        rng : numpy.random.Generator or None, default=None
            Random number generator. If `None`, a default generator is created.
        """
        self.rng = rng if rng is not None else np.random.default_rng()

        self.patching_strategies = patching_strategies
        self.target_extractors = target_extractors
        self.multiscale_count = multiscale_count
        self.alpha_ranges = alpha_ranges
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

        self.principal_channel = np.argmax(
            [patching.n_patches for patching in self.patching_strategies]
        ).item()

    @property
    def n_patches(self) -> int:
        """Return the number of available patches.

        Returns
        -------
        int
            Number of patches available from the principal target channel source.
        """
        return self.patching_strategies[self.principal_channel].n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """Return input image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Input image shapes for the principal target source.
        """
        return self.target_extractors[self.principal_channel].shapes

    @property
    def target_shapes(self) -> Sequence[Sequence[int]]:
        """Return target image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Target image shapes for the principal target source.
        """
        return self.target_extractors[self.principal_channel].shapes

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], UncorrelatedPatchSpecs]:
        """Construct the synthetic input and target patch for an index.

        Parameters
        ----------
        index : int
            Dataset index for the principal target channel source.

        Returns
        -------
        input_patch : NDArray[Any]
            Synthetic input patch with axes L(Z)YX, where L is the lateral context
            input axis ordered from the native patch scale to larger context scales.
        target_patch : NDArray[Any]
            Target patch with axes C(Z)YX.
        patch_spec : UncorrelatedPatchSpecs
            Patch specification containing one location per target channel source.
        """
        n_channels = len(self.patching_strategies)
        n_patches = [
            patching_strategy.n_patches
            for patching_strategy in self.patching_strategies
        ]
        indices = self.rng.integers(0, n_patches, n_channels).tolist()
        indices[self.principal_channel] = index
        patch_specs = [
            strategy.get_patch_spec(i)
            for strategy, i in zip(self.patching_strategies, indices, strict=True)
        ]

        patch, uncorr_patch_specs = _construct_uncorrelated_patch(
            self.target_extractors,
            patch_specs,
            channels=None,
            principal_channel=self.principal_channel,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch, target_patch = _create_input_target(
            patch, self.alpha_ranges, self.rng
        )
        return input_patch, target_patch, uncorr_patch_specs

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Return the principal input without lateral context.

        Parameters
        ----------
        input_patch : NDArray[Any]
            Input patch with axes L(Z)YX, where L is the lateral context input axis.

        Returns
        -------
        NDArray[Any]
            Principal input with axes C(Z)YX.
        """
        return input_patch[[0]]

    def get_input_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata:
        """Return metadata for the input image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Uncorrelated patch specification identifying all channel source images.

        Returns
        -------
        ImageMetadata
            Metadata for the principal input image with additional per-channel source
            metadata.
        """
        if not is_uncorrelated_specs(patch_spec):
            raise TypeError(
                # TODO: improve msg
                "Only uncorrelated patches are supported for this mode of MicroSplit "
                "training."
            )
        metadata = _get_uncorrelated_metadata(self.target_extractors, patch_spec)
        return metadata

    def get_target_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata | None:
        """Return metadata for the target image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Uncorrelated patch specification identifying all channel source images.

        Returns
        -------
        ImageMetadata
            Metadata for the principal target image with additional per-channel source
            metadata.
        """
        if not is_uncorrelated_specs(patch_spec):
            raise TypeError(
                # TODO: improve msg
                "Only uncorrelated patches are supported for this mode of MicroSplit "
                "training."
            )
        metadata = _get_uncorrelated_metadata(self.target_extractors, patch_spec)
        return metadata


# real target image and input images
class MsT3PatchConstructor(PatchConstructor):
    """Construct MicroSplit patches from paired real input and target images.

    Parameters
    ----------
    patching_strategy : Patching
        Strategy that maps dataset indices to patch specifications.
    input_extractor : PatchExtractor
        Extractor for real input images.
    target_extractor : PatchExtractor
        Extractor for target images.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}
        Padding mode used when lateral context extends beyond image boundaries.
    """

    def __init__(
        self,
        patching_strategy: Patching,
        input_extractor: PatchExtractor[Any],
        target_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
    ):
        """Initialize the training mode III patch constructor.

        Parameters
        ----------
        patching_strategy : Patching
            Strategy that maps dataset indices to patch specifications.
        input_extractor : PatchExtractor
            Extractor for real input images.
        target_extractor : PatchExtractor
            Extractor for target images.
        multiscale_count : int
            Number of lateral context inputs.
        padding_mode : {"reflect", "wrap"}
            Padding mode used when lateral context extends beyond image boundaries.
        """
        self.patching_strategy = patching_strategy
        self.input_extractor = input_extractor
        self.target_extractor = target_extractor
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

    @property
    def n_patches(self) -> int:
        """Return the number of available patches.

        Returns
        -------
        int
            Number of patches available from the patching strategy.
        """
        return self.patching_strategy.n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """Return input image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Input image shapes.
        """
        return self.input_extractor.shapes

    @property
    def target_shapes(self) -> Sequence[Sequence[int]]:
        """Return target image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Target image shapes.
        """
        return self.target_extractor.shapes

    def construct_patch(
        self, index: int
    ) -> tuple[NDArray[Any], NDArray[Any], PatchSpecs]:
        """Construct the real input and target patch for an index.

        Parameters
        ----------
        index : int
            Dataset index to map to a patch specification.

        Returns
        -------
        input_patch : NDArray[Any]
            Real input patch with axes L(Z)YX, where L is the lateral context input
            axis ordered from the native patch scale to larger context scales.
        target_patch : NDArray[Any]
            Target patch with axes C(Z)YX.
        patch_spec : PatchSpecs
            Patch specification used to extract the patches.
        """
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch = self.input_extractor.extract_patch(**patch_spec)
        input_patch = _extract_lc_patch(
            self.input_extractor,
            **patch_spec,
            channels=None,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch = input_patch.squeeze(0)  # output is CL(Z)YX
        target_patch = self.target_extractor.extract_patch(**patch_spec)
        return input_patch, target_patch, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Return the principal input without lateral context.

        Parameters
        ----------
        input_patch : NDArray[Any]
            Input patch with axes L(Z)YX, where L is the lateral context input axis.

        Returns
        -------
        NDArray[Any]
            Principal input with axes C(Z)YX.
        """
        return input_patch[[0]]

    def get_input_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata:
        """Return metadata for the input image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification identifying the input image.

        Returns
        -------
        ImageMetadata
            Metadata for the input image.
        """
        data_idx = patch_spec["data_idx"]
        image_stack = self.input_extractor.image_stacks[data_idx]
        return get_image_metadata(image_stack)

    def get_target_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata | None:
        """Return metadata for the target image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification identifying the target image.

        Returns
        -------
        ImageMetadata
            Metadata for the target image.
        """
        data_idx = patch_spec["data_idx"]
        image_stack = self.target_extractor.image_stacks[data_idx]
        return get_image_metadata(image_stack)


class MsPredPatchConstructor(PatchConstructor):
    """Construct MicroSplit prediction patches.

    Parameters
    ----------
    patching_strategy : TiledPatching
        Strategy that maps dataset indices to tiled patch specifications.
    input_extractor : PatchExtractor
        Extractor for prediction input images.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}
        Padding mode used when lateral context extends beyond image boundaries.
    """

    # prediction - input only
    def __init__(
        self,
        patching_strategy: Patching,
        input_extractor: PatchExtractor[Any],
        multiscale_count: int,
        padding_mode: Literal["reflect", "wrap"],
    ):
        """Initialize the prediction patch constructor.

        Parameters
        ----------
        patching_strategy : TiledPatching
            Strategy that maps dataset indices to tiled patch specifications.
        input_extractor : PatchExtractor
            Extractor for prediction input images.
        multiscale_count : int
            Number of lateral context inputs.
        padding_mode : {"reflect", "wrap"}
            Padding mode used when lateral context extends beyond image boundaries.
        """
        self.patching_strategy = patching_strategy
        self.input_extractor = input_extractor
        self.multiscale_count = multiscale_count
        self.padding_mode: Literal["reflect", "wrap"] = padding_mode

    @property
    def n_patches(self) -> int:
        """Return the number of available patches.

        Returns
        -------
        int
            Number of patches available from the tiled patching strategy.
        """
        return self.patching_strategy.n_patches

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """Return input image shapes.

        Returns
        -------
        Sequence[Sequence[int]]
            Input image shapes.
        """
        return self.input_extractor.shapes

    @property
    def target_shapes(self) -> None:
        """Return target image shapes, if targets exist.

        Returns
        -------
        None
            Prediction datasets do not have target images.
        """
        return None

    def construct_patch(self, index: int) -> tuple[NDArray[Any], None, PatchSpecs]:
        """Construct the input patch for prediction.

        Parameters
        ----------
        index : int
            Dataset index to map to a tile specification.

        Returns
        -------
        input_patch : NDArray[Any]
            Prediction input patch with axes L(Z)YX, where L is the lateral context
            input axis ordered from the native patch scale to larger context scales.
        target_patch : None
            Prediction datasets do not have target patches.
        patch_spec : TileSpecs
            Tile specification used to extract the patch.
        """
        patch_spec = self.patching_strategy.get_patch_spec(index)
        input_patch = _extract_lc_patch(
            self.input_extractor,
            data_idx=patch_spec["data_idx"],
            sample_idx=patch_spec["sample_idx"],
            coords=patch_spec["coords"],
            patch_size=patch_spec["patch_size"],
            channels=None,
            multiscale_count=self.multiscale_count,
            padding_mode=self.padding_mode,
        )
        input_patch = input_patch.squeeze(0)  # output is CL(Z)YX
        return input_patch, None, patch_spec

    def get_principal_input(self, input_patch: NDArray[Any]) -> NDArray[Any]:
        """Return the principal input without lateral context.

        Parameters
        ----------
        input_patch : NDArray[Any]
            Input patch with axes L(Z)YX, where L is the lateral context input axis.

        Returns
        -------
        NDArray[Any]
            Principal input with axes C(Z)YX.
        """
        return input_patch[[0]]

    def get_input_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata:
        """Return metadata for the input image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification identifying the input image.

        Returns
        -------
        ImageMetadata
            Metadata for the input image.
        """
        data_idx = patch_spec["data_idx"]
        image_stack = self.input_extractor.image_stacks[data_idx]
        return get_image_metadata(image_stack)

    def get_target_image_metadata(self, patch_spec: PatchSpecs) -> ImageMetadata | None:
        """Return metadata for the target image.

        Parameters
        ----------
        patch_spec : PatchSpecs
            Patch specification. It is accepted to satisfy the protocol but is unused
            because prediction datasets do not have targets.

        Returns
        -------
        None
            Prediction datasets do not have target images.
        """
        return None


def _sample_alphas(
    alpha_ranges: Sequence[tuple[float, float]] | None,
    n_channels: int,
    rng: np.random.Generator,
) -> NDArray[Any]:
    """Sample channel mixing weights.

    Parameters
    ----------
    alpha_ranges : Sequence[tuple[float, float]] or None
        Sampling ranges for channel mixing weights. If `None`, every channel gets a
        fixed weight of `1 / n_channels`.
    n_channels : int
        Number of channel weights to sample.
    rng : numpy.random.Generator
        Random number generator used when `alpha_ranges` is provided.

    Returns
    -------
    NDArray[Any]
        Channel mixing weights with axes C.
    """
    if alpha_ranges is None:
        return np.array([1 / n_channels for _ in range(n_channels)])
    else:
        low = tuple(r[0] for r in alpha_ranges)
        high = tuple(r[1] for r in alpha_ranges)
        return rng.uniform(low, high)


def _create_input_target(
    patch: NDArray[Any],
    alpha_ranges: Sequence[tuple[float, float]] | None,
    rng: np.random.Generator,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Create synthetic MicroSplit input and target patches.

    Parameters
    ----------
    patch : NDArray[Any]
        Lateral context patch with axes CL(Z)YX, where L is the lateral context input
        axis ordered from the native patch scale to larger context scales.
    alpha_ranges : Sequence[tuple[float, float]] or None
        Sampling ranges for channel mixing weights. If `None`, every channel gets a
        fixed weight of `1 / n_channels`.
    rng : numpy.random.Generator
        Random number generator used when `alpha_ranges` is provided.

    Returns
    -------
    input_patch : NDArray[Any]
        Synthetic input patch with axes L(Z)YX, where L is the lateral context input
        axis.
    target_patch : NDArray[Any]
        Target patch with axes C(Z)YX.
    """
    alphas = _sample_alphas(alpha_ranges, patch.shape[0], rng)
    alpha_broadcast = alphas.reshape(len(alphas), *(1,) * (len(patch.shape) - 1))
    # weight channels by alphas then sum on the channel axis
    # input dims will be L(Z)YX
    input_patch = (alpha_broadcast * patch).sum(axis=0)
    target_patch = patch[:, 0, ...]  # first L patch
    return input_patch, target_patch


def _construct_uncorrelated_patch(
    extractor: PatchExtractor[Any] | Sequence[PatchExtractor[Any]],
    patch_specs: Sequence[PatchSpecs],
    channels: Sequence[int] | None,
    principal_channel: int,
    multiscale_count: int,
    padding_mode: Literal["reflect", "wrap"] = "reflect",
) -> tuple[NDArray[Any], UncorrelatedPatchSpecs]:
    """Construct a patch from channel patches sampled at different locations.

    Parameters
    ----------
    extractor : PatchExtractor or Sequence[PatchExtractor]
        Extractor used to sample all channels when a single extractor is provided, or
        one extractor per channel when a sequence is provided.
    patch_specs : Sequence[PatchSpecs]
        Patch specification for each sampled channel.
    channels : Sequence[int] or None
        Channel indices to extract when `extractor` is a single patch extractor. If
        `None`, channel indices are inferred from the order of `patch_specs`.
    principal_channel : int
        Index of the channel whose patch specification is used as the primary region.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}, default="reflect"
        Padding mode used when lateral context extends beyond image boundaries.

    Returns
    -------
    patch : NDArray[Any]
        Uncorrelated lateral context patch with axes CL(Z)YX, where L is the lateral
        context input axis ordered from the native patch scale to larger context
        scales.
    patch_spec : UncorrelatedPatchSpecs
        Patch specification containing all sampled channel locations.
    """
    # TODO: cannot use channels with sequence of extractors

    n_channels = len(patch_specs)
    patch_size = patch_specs[0]["patch_size"]
    patch = np.zeros((n_channels, multiscale_count, *patch_size))
    for c_idx in range(n_channels):
        if isinstance(extractor, PatchExtractor):
            original_c_idx = c_idx if channels is None else channels[c_idx]
            patch[[c_idx]] = _extract_lc_patch(
                extractor,
                **patch_specs[c_idx],
                channels=[original_c_idx],
                multiscale_count=multiscale_count,
                padding_mode=padding_mode,
            )
        else:
            patch[[c_idx]] = _extract_lc_patch(
                extractor[c_idx],
                **patch_specs[c_idx],
                # TODO: guard that sequence of extractors each have n_channels==1
                channels=None,
                multiscale_count=multiscale_count,
                padding_mode=padding_mode,
            )
    uncorr_patch_specs = UncorrelatedPatchSpecs(
        **patch_specs[principal_channel],
        principal_channel=principal_channel,
        all_data_idx=[ps["data_idx"] for ps in patch_specs],
        all_sample_idx=[ps["sample_idx"] for ps in patch_specs],
        all_coords=[ps["coords"] for ps in patch_specs],
    )
    return patch, uncorr_patch_specs


def _extract_lc_patch(
    extractor: PatchExtractor[ImageStack],
    data_idx: int,
    sample_idx: int,
    channels: Sequence[int] | None,
    coords: Sequence[int],
    patch_size: Sequence[int],
    multiscale_count: int,
    padding_mode: Literal["reflect", "wrap"] = "reflect",
) -> NDArray[Any]:
    """Extract a patch with lateral context inputs.

    Parameters
    ----------
    extractor : PatchExtractor
        Extractor used to sample the patch.
    data_idx : int
        Index of the image stack to sample.
    sample_idx : int
        Index of the sample within the image stack.
    channels : Sequence[int] or None
        Channel indices to extract. If `None`, all channels are extracted.
    coords : Sequence[int]
        Spatial start coordinates of the native-scale patch.
    patch_size : Sequence[int]
        Spatial size of the native-scale patch.
    multiscale_count : int
        Number of lateral context inputs.
    padding_mode : {"reflect", "wrap"}, default="reflect"
        Padding mode used when lateral context extends beyond image boundaries.

    Returns
    -------
    NDArray[Any]
        Lateral context patch with axes CL(Z)YX, where L is the lateral context input
        axis ordered from the native patch scale to larger context scales.
    """
    shape = extractor.image_stacks[data_idx].data_shape
    spatial_shape = shape[2:]
    n_channels = shape[1] if channels is None else len(channels)

    center = np.array(coords) + np.array(patch_size) // 2
    final_lc_patch_size = np.array(patch_size) * (2**multiscale_count)
    final_lc_start = center - final_lc_patch_size // 2
    final_lc_end = final_lc_start + np.array(final_lc_patch_size)

    start_clipped = np.clip(
        final_lc_start, np.zeros_like(spatial_shape), np.array(spatial_shape)
    )
    end_clipped = np.clip(
        final_lc_end, np.zeros_like(spatial_shape), np.array(spatial_shape)
    )
    size_clipped = end_clipped - start_clipped

    final_lc_patch = extractor.extract_channel_patch(
        data_idx, sample_idx, channels, start_clipped, size_clipped
    )
    pad_before = start_clipped - final_lc_start
    pad_after = final_lc_end - end_clipped
    pad_width = np.concatenate(
        [
            # zeros to not pad the channel axis
            np.zeros((1, 2), dtype=int),
            np.stack([pad_before, pad_after], axis=-1),
        ]
    )
    final_lc_patch = np.pad(
        final_lc_patch,
        pad_width,
        mode=padding_mode,
    )

    patch = np.zeros((n_channels, multiscale_count, *patch_size), dtype=np.float32)
    for scale in range(multiscale_count):
        lc_patch_size = np.array(patch_size) * (2**scale)
        lc_start = final_lc_patch_size // 2 - lc_patch_size // 2
        lc_end = lc_start + lc_patch_size

        spatial_slice = [slice(s, t) for s, t in zip(lc_start, lc_end, strict=True)]
        lc_patch = final_lc_patch[..., *spatial_slice]  # type: ignore
        # TODO: test different downscaling? skimage suggests downscale_local_mean
        lc_patch = resize(lc_patch, (n_channels, *patch_size))
        patch[:, scale, ...] = lc_patch
    return patch


def _get_uncorrelated_metadata(
    target_extractor: PatchExtractor[ImageStack] | Sequence[PatchExtractor[ImageStack]],
    patch_spec: UncorrelatedPatchSpecs,
) -> ImageMetadata:
    """Return metadata for an uncorrelated MicroSplit patch.

    Parameters
    ----------
    target_extractor : PatchExtractor or Sequence[PatchExtractor]
        Extractor containing all channel sources when a single extractor is provided,
        or one extractor per channel when a sequence is provided.
    patch_spec : UncorrelatedPatchSpecs
        Patch specification containing all sampled channel locations.

    Returns
    -------
    ImageMetadata
        Metadata for the principal channel image with additional metadata for all
        sampled channel images.
    """
    if isinstance(target_extractor, PatchExtractor):
        image_stacks = [
            target_extractor.image_stacks[data_idx]
            for data_idx in patch_spec["all_data_idx"]
        ]
    else:  # sequence
        image_stacks = [
            extractor.image_stacks[data_idx]
            for extractor, data_idx in zip(
                target_extractor, patch_spec["all_data_idx"], strict=True
            )
        ]
    principal_image_stack = image_stacks[patch_spec["principal_channel"]]
    all_sources = [str(image_stack.source) for image_stack in image_stacks]
    all_data_shapes = [image_stack.data_shape for image_stack in image_stacks]
    all_original_data_shapes = [
        image_stack.original_data_shape for image_stack in image_stacks
    ]

    metadata = get_image_metadata(principal_image_stack)
    metadata["additional_metadata"]["all_sources"] = all_sources
    metadata["additional_metadata"]["all_data_shapes"] = all_data_shapes
    metadata["additional_metadata"][
        "all_original_data_shapes"
    ] = all_original_data_shapes
    return metadata
