from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from skimage.transform import resize

from ..image_stack import ImageStack


class PatchConstructor(Protocol):
    """
    A callable that modifies how patches are constructed in the PatchExtractor.

    This protocol defines the signature of a callable that is passed as an argument to
    the `PatchExtractor`. It can be used to modify how patches are constructed, for
    example creating patches with multiple lateral context levels for MicroSplit.
    """

    def __call__(
        self,
        image_stack: ImageStack,
        sample_idx: int,
        channels: Sequence[int] | None,  # `channels = None` to select all channels
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[Any]:
        """
        Parameters
        ----------
        image_stack: ImageStack
            The image stack to construct a patch from.
        sample_idx: int
            Sample index. The first dimension of the image data will be indexed at this
            value.
        coords: Sequence of int
            The coordinates that define the start of a patch.
        patch_size: Sequence of int
            The size of the patch in each spatial dimension.

        Returns
        -------
        numpy.ndarray
            The patch.
        """
        ...


def default_patch_constr(
    image_stack: ImageStack,
    sample_idx: int,
    channels: Sequence[int] | None,  # `channels = None` to select all channels
    coords: Sequence[int],
    patch_size: Sequence[int],
) -> NDArray[Any]:
    return image_stack.extract_channel_patch(
        sample_idx=sample_idx,
        channels=channels,
        coords=coords,
        patch_size=patch_size,
    )


# closure to create constructor funcs with particular multiscale_count and padding mode
def lateral_context_patch_constr(
    # TODO: will we stick with this as the parameter name
    multiscale_count: int,
    # TODO: add other modes?
    padding_mode: Literal["reflect", "wrap"],
) -> PatchConstructor:
    """
    Create a lateral context `PatchConstructor` for MicroSplit.

    Parameters
    ----------
    multiscale_count : int
        The number of multiscale inputs that will be created including the original
        image size.
    padding_mode : {"reflect", "wrap"}
        How lateral context inputs will be padded at the edge of the image. See
        [`numpy.pad`](https://numpy.org/devdocs/reference/generated/numpy.pad.html) for
        more information.

    Returns
    -------
    PatchConstructor
        The patch constructor function. It will return patches with the dimensions
        (C, L, (Z), Y, X) where L will be equal to `multiscale_count`, C is the number
        of channels in the image, and (Z), Y, X are the patch size.
    """

    def constructor_func(
        image_stack: ImageStack,
        sample_idx: int,
        channels: Sequence[int] | None,  # `channels = None` to select all channels
        coords: Sequence[int],
        patch_size: Sequence[int],
    ) -> NDArray[Any]:
        if channels is not None and len(channels) > 1:
            raise NotImplementedError(
                "Selecting multiple channels is currently not implemented for lateral "
                "context patches. Select a single channel or pass `channels=None` to "
                "select all channels."
            )

        shape = image_stack.data_shape
        spatial_shape = shape[2:]
        n_channels = shape[1] if channels is None else 1

        # There will now be an additional lc dimension,
        # this has to be handled correctly by the dataset
        # TODO: maybe we want to limit this constructor to only images with 1 channel
        #   then we can put LCs in the channel dimension
        #   but not sure if this artificially limits potential use-cases
        patch = np.zeros((n_channels, multiscale_count, *patch_size))
        for scale in range(multiscale_count):
            lc_patch_size = np.array(patch_size) * (2**scale)
            lc_start = np.array(coords) + np.array(patch_size) // 2 - lc_patch_size // 2
            lc_end = lc_start + np.array(lc_patch_size)

            start_clipped = np.clip(
                lc_start, np.zeros_like(spatial_shape), np.array(spatial_shape)
            )
            end_clipped = np.clip(
                lc_end, np.zeros_like(spatial_shape), np.array(spatial_shape)
            )
            size_clipped = end_clipped - start_clipped

            lc_patch = image_stack.extract_channel_patch(
                sample_idx, channels, start_clipped, size_clipped
            )
            pad_before = start_clipped - lc_start
            pad_after = lc_end - end_clipped
            pad_width = np.concat(
                [
                    # zeros to not pad the channel axis
                    np.zeros((1, 2), dtype=int),
                    np.stack([pad_before, pad_after], axis=-1),
                ]
            )
            lc_patch = np.pad(
                lc_patch,
                pad_width,
                mode=padding_mode,
            )
            # TODO: test different downscaling? skimage suggests downscale_local_mean
            lc_patch = resize(lc_patch, (n_channels, *patch_size))
            patch[:, scale, ...] = lc_patch
        return patch

    return constructor_func
