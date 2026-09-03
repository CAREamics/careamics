"""Type for data produced by the dataset and propagated through models."""

from collections.abc import Sequence
from typing import Any, Generic, Literal, NamedTuple, Union

from numpy.typing import NDArray

from .patching import (
    RegionSpecs,
)


class ImageRegionData(NamedTuple, Generic[RegionSpecs]):
    """
    Data structure for arrays produced by the dataset and propagated through models.

    An ImageRegionData may be a patch during training/validation, a tile during
    prediction with tiling, or a whole image during prediction without tiling.

    `canonical_shape` may not correspond to the shape of the original data if a subset
    of the channels has been requested, in which case the channel dimension may
    be smaller than that of the original data and only correspond to the requested
    number of channels.

    ImageRegionData may be collated in batches during training by the DataLoader.
    Description of the fields is given for the uncollated case (non-batched).

    Canonical space refers to the SC(Z)YX space, while original space is the original
    axes order of the data.
    """

    data: NDArray
    """Patch, tile or image in canonical space."""

    source: Union[str, Literal["array"]]
    """Source of the data, e.g. file path, zarr URI, or "array" for in-memory arrays."""

    canonical_shape: Sequence[int]
    """Shape of the source image in canonical space. Axis order may differ from the
    source due to reshaping to canonical space. Channel dimension may differ from
    if channels were subsetted. `data` may be a patch or a tile from the source data,
    and can therefore have a different shape from `canonical_shape`."""

    dtype: str  # dtype should be str for collate
    """Data type of the source image as a string."""

    original_axes: str
    """Axes of the source image in the original space."""

    target_axes: str
    """Axes of the target array corresponding to the source image. If no target was
    used during training, this should be set to `original_axes`. Target axes may differ
    from `original_axes` depending on the task.
    """

    original_shape: Sequence[int]
    """Original shape of the source image before reshaping in canonical space and may
    include channel subsetting."""

    region_spec: RegionSpecs  # PatchSpecs or subclasses, e.g. TileSpecs
    """Specifications of the region within the source image from where `data` is
    extracted. Of type PatchSpecs during training/validation and prediction without
    tiling, and TileSpecs during prediction with tiling.
    """

    additional_metadata: dict[str, Any]
    """Additional metadata to be stored with the image region. Used to store chunk and
    shard information for zarr image stacks."""

    @classmethod
    def from_model_output(
        cls, input_region: "ImageRegionData", output_data: NDArray
    ) -> "ImageRegionData":
        """
        Create an ImageRegionData from the model output and the input region.

        Parameters
        ----------
        input_region : ImageRegionData
            The input region that was passed to the model.
        output_data : NDArray
            The output data from the model.

        Returns
        -------
        ImageRegionData
            The output region containing the model predictions.
        """
        return cls(
            data=output_data,
            source=input_region.source,
            canonical_shape=input_region.canonical_shape,
            dtype=input_region.dtype,
            original_axes=input_region.original_axes,
            target_axes=input_region.target_axes,
            original_shape=input_region.original_shape,
            region_spec=input_region.region_spec,
            additional_metadata=input_region.additional_metadata,
        )
