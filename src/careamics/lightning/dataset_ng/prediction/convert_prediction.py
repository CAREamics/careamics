"""Module containing functions to convert prediction outputs to desired form."""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData

from .stitch_prediction import group_tiles_by_key, stitch_prediction

if TYPE_CHECKING:
    from torch import Tensor


def _decollate_batch_dict(
    batched_dict: "dict[str, list | Tensor]",
    index: int,
) -> dict[str, int | tuple[int, ...]]:
    """
    Decollate element `index` from a batched_dict.

    This method is only compatible with integer elements.

    Parameters
    ----------
    batched_dict : dict of {str: list or Tensor}
        Batch dictionary where each value is a list of elements of length B or a
        Tensor of shape (B,).
    index : int
        Index of the element to extract.

    Returns
    -------
    dict of {str: int | tuple[int, ...]}
        Dictionary of the `index` element in the collated batch.
    """
    item_dict = {
        key: (
            # cast to int otherwise we have Tensor scalars
            # TODO for additional types (e.g. axes in additional_metadata), we will need
            # to handle it differently
            tuple(int(value[idx][index]) for idx in range(len(value)))
            if isinstance(value, list)
            else int(value[index])
        )  # handles tensor (1D) vs list of 1D tensors (2D)
        for key, value in batched_dict.items()
    }

    return item_dict


def decollate_image_region_data(
    batch: ImageRegionData,
) -> list[ImageRegionData]:
    """
    Decollate a batch of `ImageRegionData` into a list of `ImageRegionData`.

    Input batch has the following structure:
    - data: (B, C, (Z), Y, X) numpy.ndarray
    - source: sequence of str, length B
    - data_shape: sequence of tuple of int, each tuple being of length B
    - dtype: list of numpy.dtype, length B
    - axes: list of str, length B
    - region_spec: dict of {str: sequence}, each sequence being of length B
    - additional_metadata: dict of {str: Any}, each sequence being of length B

    Parameters
    ----------
    batch : ImageRegionData
        Batch of `ImageRegionData`.

    Returns
    -------
    list of ImageRegionData
        List of `ImageRegionData`.
    """
    batch_size = batch.data.shape[0]
    decollated: list[ImageRegionData] = []
    for i in range(batch_size):
        # unpack region spec irrespective of whether it is a PatchSpecs or TileSpecs
        region_spec = _decollate_batch_dict(batch.region_spec, i)

        # handle additional metadata
        # currently only zarr chunks and shards may be stored there, as tuples.
        # TODO if additional metadata becomes used for anything else, this function
        # call may not be appropriate anymore.
        additional_metadata = _decollate_batch_dict(batch.additional_metadata, i)

        # data shape
        assert isinstance(batch.data_shape, list)
        data_shape = tuple(int(dim[i]) for dim in batch.data_shape)

        image_region = ImageRegionData(
            data=batch.data[i],  # discard batch dimension
            source=batch.source[i],
            dtype=batch.dtype[i],
            data_shape=data_shape,
            axes=batch.axes[i],
            region_spec=region_spec,  # type: ignore
            additional_metadata=additional_metadata,
        )
        decollated.append(image_region)

    return decollated


def combine_samples(
    predictions: list[ImageRegionData],
) -> tuple[list[NDArray], list[str]]:
    """
    Combine predictions by `data_idx`.

    Images are first grouped by their `data_idx` found in their `region_spec`, then
    sorted by ascending `sample_idx` before being stacked along the `S` dimension.

    Parameters
    ----------
    predictions : list of ImageRegionData
        List of `ImageRegionData`.

    Returns
    -------
    list of numpy.ndarray
        List of combined predictions, one per unique `data_idx`.
    list of str
        List of sources, one per unique `data_idx`.
    """
    # group predictions by data idx
    grouped_prediction: dict[int, list[ImageRegionData]] = group_tiles_by_key(
        predictions, key="data_idx"
    )

    # sort predictions by sample idx
    combined_predictions: list[NDArray] = []
    combined_sources: list[str] = []
    for data_idx in sorted(grouped_prediction.keys()):
        image_regions = grouped_prediction[data_idx]
        combined_sources.append(image_regions[0].source)

        # sort by sample idx
        image_regions.sort(key=lambda x: x.region_spec["sample_idx"])

        # remove singleton dims and stack along S axis
        combined_data = np.stack([img.data.squeeze() for img in image_regions], axis=0)
        combined_predictions.append(combined_data)

    return combined_predictions, combined_sources


def convert_prediction(
    predictions: list[ImageRegionData],
    tiled: bool,
) -> tuple[list[NDArray], list[str]]:
    """
    Convert the Lightning trainer outputs to the desired form.

    This method allows decollating batches and stitching back together tiled
    predictions.

    If the `source` of all predictions is "array" (see `InMemoryImageStack`), then the
    returned sources list will be empty.

    Parameters
    ----------
    predictions : list[ImageRegionData]
        Output from `Trainer.predict`, list of batches.
    tiled : bool
        Whether the predictions are tiled.

    Returns
    -------
    list of numpy.ndarray
        List of arrays with the axes SC(Z)YX.
    list of str
        List of sources, one per output or empty if all equal to `array`.
    """
    # decollate batches
    decollated_predictions: list[ImageRegionData] = []
    for batch in predictions:
        decollated_batch = decollate_image_region_data(batch)
        decollated_predictions.extend(decollated_batch)

    if not tiled and "total_tiles" in decollated_predictions[0].region_spec:
        raise ValueError(
            "Predictions contain `total_tiles` in region_spec but `tiled` is set to "
            "False."
        )

    if tiled:
        predictions_output, sources = stitch_prediction(decollated_predictions)
    else:
        predictions_output, sources = combine_samples(decollated_predictions)

    if set(sources) == {"array"}:
        sources = []

    return predictions_output, sources
