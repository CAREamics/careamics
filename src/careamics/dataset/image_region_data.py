"""Type for data produced by the dataset and propagated through models."""

from collections.abc import Sequence
from typing import Any, Generic, Literal, NamedTuple, Union

from numpy.typing import NDArray

from .patching_strategies import (
    RegionSpecs,
)


class ImageRegionData(NamedTuple, Generic[RegionSpecs]):
    """
    Data structure for arrays produced by the dataset and propagated through models.

    An ImageRegionData may be a patch during training/validation, a tile during
    prediction with tiling, or a whole image during prediction without tiling.

    `data_shape` may not correspond to the shape of the original data if a subset
    of the channels has been requested, in which case the channel dimension may
    be smaller than that of the original data and only correspond to the requested
    number of channels.

    ImageRegionData may be collated in batches during training by the DataLoader. In
    that case:
    - data: arrays are collated into NDArray of shape (B,C,Z,Y,X)
    - source: list of str, length B
    - data_shape: list of tuples of int, each tuple being of length B and representing
        the shape of the original images in the corresponding dimension
    - dtype: list of str, length B
    - axes: list of str, length B
    - region_spec: dict of {str: sequence}, each sequence being of length B
    - additional_metadata: list of dict

    Description of the fields is given for the uncollated case (non-batched).
    """

    data: NDArray
    """Patch, tile or image in C(Z)YX format."""

    source: Union[str, Literal["array"]]
    """Source of the data, e.g. file path, zarr URI, or "array" for in-memory arrays."""

    data_shape: Sequence[int]
    """Shape of the original image in (SCZ)YX format and order. If channels are
    subsetted, the channel dimension corresponds to the number of requested channels."""

    dtype: str  # dtype should be str for collate
    """Data type of the original image as a string."""

    axes: str
    """Axes of the original data array. SCTZYX dimensions are allowed in any order."""

    original_data_shape: Sequence[int]
    """Original shape of the data before any reshaping."""

    region_spec: RegionSpecs  # PatchSpecs or subclasses, e.g. TileSpecs
    """Specifications of the region within the original image from where `data` is
    extracted. Of type PatchSpecs during training/validation and prediction without
    tiling, and TileSpecs during prediction with tiling.
    """

    additional_metadata: dict[str, Any]
    """Additional metadata to be stored with the image region. Currently used to store
    chunk and shard information for zarr image stacks."""
