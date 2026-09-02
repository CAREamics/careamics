"""Utilities for decollating prediction batches."""

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from careamics.dataset.image_region_data import ImageRegionData
from careamics.dataset.patching import PatchSpecs


def _is_scalar(value: Any) -> bool:
    """Return whether a value is scalar-like.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True if the value behaves like a scalar.
    """
    return isinstance(value, str | bytes | int | float | bool | np.generic)


def _is_string_scalar(value: Any) -> bool:
    """Return whether a value is a string-like scalar.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True if the value is a string-like scalar.
    """
    return isinstance(value, str | bytes)


def _is_tensor_like(value: Any) -> bool:
    """Return whether a value behaves like a tensor.

    Parameters
    ----------
    value : Any
        Value to inspect.

    Returns
    -------
    bool
        True if the value exposes tensor-like indexing.
    """
    return hasattr(value, "ndim") and hasattr(value, "__getitem__")


def _extract_tensor_item(value: Any) -> Any:
    """Convert a tensor-like scalar to a Python scalar when possible.

    Parameters
    ----------
    value : Any
        Tensor-like or scalar value.

    Returns
    -------
    Any
        Extracted Python scalar or the original value.
    """
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    return value


def _decollate_batch_value(value: Any, index: int) -> Any:
    """Decollate one value from a collated batch object.

    Parameters
    ----------
    value : Any
        Collated batch value.
    index : int
        Index of the sample to extract.

    Returns
    -------
    Any
        Decollated value for one sample.
    """
    if isinstance(value, Mapping):
        return {key: _decollate_batch_value(item, index) for key, item in value.items()}

    if isinstance(value, list):
        if len(value) > index and all(_is_scalar(item) for item in value):
            return value[index]

        extracted = [_decollate_batch_value(item, index) for item in value]

        if all(_is_scalar(item) for item in extracted):
            if any(_is_string_scalar(item) for item in extracted):
                return extracted
            return tuple(extracted)

        return extracted

    if isinstance(value, tuple):
        if len(value) > index and all(_is_scalar(item) for item in value):
            return value[index]

        extracted_tuple = tuple(_decollate_batch_value(item, index) for item in value)

        if all(_is_scalar(item) for item in extracted_tuple):
            if any(_is_string_scalar(item) for item in extracted_tuple):
                return list(extracted_tuple)
            return extracted_tuple

        return list(extracted_tuple)

    if _is_tensor_like(value):
        return _extract_tensor_item(value[index])

    return value


def _decollate_batch_dict(
    batched_dict: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Decollate element `index` from a batched dictionary.

    Parameters
    ----------
    batched_dict : dict[str, Any]
        Batch dictionary where each value is a collated batch object.
    index : int
        Index of the element to extract.

    Returns
    -------
    dict[str, Any]
        Dictionary of the `index` element in the collated batch.
    """
    return {
        key: _decollate_batch_value(value, index) for key, value in batched_dict.items()
    }


def decollate_image_region_data(
    batch: ImageRegionData,
) -> list[ImageRegionData[PatchSpecs]]:
    """Decollate a batch of `ImageRegionData`.

    Parameters
    ----------
    batch : ImageRegionData
        Batch of `ImageRegionData`.

    Returns
    -------
    list[ImageRegionData]
        List of `ImageRegionData`.
    """
    batch_size = batch.data.shape[0]
    decollated: list[ImageRegionData[PatchSpecs]] = []
    for i in range(batch_size):
        region_spec = cast(PatchSpecs, _decollate_batch_dict(batch.region_spec, i))
        additional_metadata = _decollate_batch_dict(batch.additional_metadata, i)

        assert isinstance(batch.data_shape, list)
        data_shape = tuple(int(dim[i]) for dim in batch.data_shape)

        assert isinstance(batch.original_data_shape, list)
        original_data_shape = tuple(int(dim[i]) for dim in batch.original_data_shape)

        image_region = ImageRegionData(
            data=batch.data[i],
            source=batch.source[i],
            dtype=batch.dtype[i],
            data_shape=data_shape,
            axes=batch.axes[i],
            target_axes=batch.target_axes[i],
            region_spec=region_spec,
            additional_metadata=additional_metadata,
            original_data_shape=original_data_shape,
        )
        decollated.append(image_region)

    return decollated
