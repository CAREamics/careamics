from collections.abc import Sequence
from typing import Any, TypedDict

from careamics.dataset.image_stack import ImageStack, ZarrImageStack


class ImageMetadata(TypedDict):
    source: str
    dtype: str
    data_shape: Sequence[int]
    original_data_shape: Sequence[int]
    additional_metadata: dict[str, Any]


# TODO: perhaps this should be a method on the image stacks themselves
def get_image_metadata(image_stack: ImageStack) -> ImageMetadata:
    # additional metadata for zarr image stacks
    if isinstance(image_stack, ZarrImageStack):
        additional_metadata: dict[str, Any] = {
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
