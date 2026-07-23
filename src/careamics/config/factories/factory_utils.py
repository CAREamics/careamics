"""Utility to handle augmentation configuration."""

from collections.abc import Sequence
from typing import Literal

from careamics.config.augmentations import (
    SPATIAL_TRANSFORMS_UNION,
    XYFlipConfig,
    XYRandomRotate90Config,
)

from .data_factory import list_spatial_augmentations


def validate_input_channels(
    axes: str,
    channels: Sequence[int] | None,
    n_channels: int | None,
    attr_name: str = "n_channels",
) -> None:
    """Validate channel dimensions.

    Parameters
    ----------
    axes : str
        Axes of the data (e.g. SYX).
    channels : Sequence[int] | None
        List of channels to use. If `None`, all channels are used.
    n_channels : int | None
        Number of channels (in and out). If `channels` is specified, then the number
        of channels is inferred from its length and this parameter is ignored.
    attr_name : str
        Name of the attribute to use in error messages (default: "n_channels").

    Raises
    ------
    ValueError
        If `channels` is specified but `axes` does not include "C".
        If `channels` is specified but is empty.
        If `channels` is specified but its length does not match `n_channels`.
        If `n_channels` is specified but `axes` does not include "C".
    """
    channels_present = "C" in axes

    if channels is not None and len(channels) == 0:
        raise ValueError("`channels` cannot not be empty.")

    if channels is not None and not channels_present:
        raise ValueError("`channels` can only be specified when `axes` includes 'C'.")

    if channels_present and (n_channels is None and channels is None):
        raise ValueError(
            f"`{attr_name}` or `channels` must be specified when using channels."
        )

    if not channels_present and n_channels is not None and n_channels > 1:
        raise ValueError(
            f"C is not present in the axes, but number of channels is specified "
            f"(got {n_channels} channel)."
        )

    if n_channels is not None and channels is not None and n_channels != len(channels):
        raise ValueError(
            f"Number of channels ({n_channels}) does not match length of "
            f"`channels` ({len(channels)}). Only specify `channels`."
        )


def assemble_augmentations(
    augmentations: Sequence[Literal["x_flip", "y_flip", "rotate_90"]] | None,
    seed: int | None = None,
) -> list[SPATIAL_TRANSFORMS_UNION]:
    """Assemble a list of augmentations.

    Parameters
    ----------
    augmentations : Sequence of {"x_flip", "y_flip", "rotate_90"} or None
        List of augmentations to apply. If `None`, all augmentations are applied.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list of augmentations
        List of augmentations to apply.
    """
    augs: list[XYFlipConfig | XYRandomRotate90Config] | None = None
    if augmentations is not None:
        augs = []

        x_flip_present = "x_flip" in augmentations
        y_flip_present = "y_flip" in augmentations
        rotate_90_present = "rotate_90" in augmentations

        if x_flip_present or y_flip_present:
            augs.append(
                XYFlipConfig(
                    flip_x=x_flip_present,
                    flip_y=y_flip_present,
                    seed=seed,
                )
            )
        if rotate_90_present:
            augs.append(XYRandomRotate90Config(seed=seed))

    return list_spatial_augmentations(augs)
