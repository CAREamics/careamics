"""Utilities for collecting image metadata from image stacks."""

from collections.abc import Sequence
from typing import Any, TypedDict

from careamics.dataset.image_stack import ImageStack, MrcImageStack, ZarrImageStack


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
        Format-specific metadata, such as chunking for zarr data.
    """

    source: str
    dtype: str
    data_shape: Sequence[int]
    original_data_shape: Sequence[int]
    additional_metadata: dict[str, Any]


# TODO: perhaps this should be a method on the image stacks themselves
# - currently Zarr decision (chunks and shards) are done here but that belongs to
#   the image stack
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
    # additional metadata for format-specific image stacks
    additional_metadata: dict[str, Any]
    if isinstance(image_stack, MrcImageStack):
        additional_metadata = image_stack.additional_metadata
    elif isinstance(image_stack, ZarrImageStack):
        additional_metadata = {
            "chunks": image_stack.chunks,
        }

        if image_stack.shards is not None:
            additional_metadata["shards"] = image_stack.shards
    else:
        additional_metadata = {}

    return {
        "source": str(image_stack.source),
        "dtype": str(image_stack.data_dtype),
        "data_shape": image_stack.data_shape,
        "original_data_shape": image_stack.original_data_shape,
        "additional_metadata": additional_metadata,
    }
