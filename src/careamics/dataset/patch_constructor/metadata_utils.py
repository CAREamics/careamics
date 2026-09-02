"""Utilities for collecting image metadata from image stacks."""

from collections.abc import Sequence
from typing import Any, TypedDict

from careamics.dataset.image_stack import ImageStack


class ImageMetadata(TypedDict):
    """Metadata describing an image stack used to create a patch.

    Attributes
    ----------
    source : str
        Source path or identifier for the image stack.
    dtype : str
        Data type of the image stack.
    data_shape : Sequence[int]
        Loaded data shape.
    original_data_shape : Sequence[int]
        Original source data shape.
    additional_metadata : dict[str, Any]
        Format-specific metadata.
    """

    source: str
    dtype: str
    data_shape: Sequence[int]
    original_data_shape: Sequence[int]
    additional_metadata: dict[str, Any]


def get_image_metadata(image_stack: ImageStack) -> ImageMetadata:
    """Return metadata for an image stack.

    Parameters
    ----------
    image_stack : ImageStack
        Image stack to describe.

    Returns
    -------
    ImageMetadata
        Metadata for the image stack.
    """
    return {
        "source": str(image_stack.source),
        "dtype": str(image_stack.data_dtype),
        "data_shape": image_stack.data_shape,
        "original_data_shape": image_stack.original_data_shape,
        "additional_metadata": dict(image_stack.additional_metadata),
    }
