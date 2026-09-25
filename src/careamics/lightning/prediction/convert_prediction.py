"""Module containing functions to convert prediction outputs to desired form."""

import numpy as np
from numpy.typing import NDArray

from careamics.dataset.image_region_data import ImageRegionData
from careamics.utils.reshape_array import restore_array

from .decollate_utils import decollate_image_region_data
from .stitch_prediction import group_tiles_by_key, stitch_prediction


def combine_samples(
    predictions: list[ImageRegionData],
    restore_shape: bool = False,
) -> tuple[list[NDArray], list[str]]:
    """
    Combine predictions by `data_idx`.

    Images are first grouped by their `data_idx` found in their `region_spec`, then
    sorted by ascending `sample_idx` before being stacked along the `S` dimension.

    Parameters
    ----------
    predictions : list of ImageRegionData
        List of `ImageRegionData`.
    restore_shape : bool, default=False
        If True, restore predictions to their original shape and dimension order.

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

        # stack along S axis (keep C dimension if present)
        # data is in C(Z)YX format, we want to stack along new S dimension
        combined_data = np.stack([img.data for img in image_regions], axis=0)

        if restore_shape:
            # get original shape info from the first image region
            original_axes = image_regions[0].axes
            target_axes = image_regions[0].target_axes
            original_data_shape = image_regions[0].original_data_shape
            combined_data = restore_array(
                combined_data, original_axes, original_data_shape, target_axes
            )

        combined_predictions.append(combined_data)

    return combined_predictions, combined_sources


def convert_prediction(
    predictions: list[ImageRegionData],
    tiled: bool,
    restore_shape: bool = False,
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
    restore_shape : bool, default=False
        If True, restore predictions to their original shape and dimension order.

    Returns
    -------
    list of numpy.ndarray
        List of arrays with the axes SC(Z)YX, or original axes if restore_shape=True.
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
        predictions_output, sources = stitch_prediction(
            decollated_predictions, restore_shape=restore_shape
        )
    else:
        predictions_output, sources = combine_samples(
            decollated_predictions, restore_shape=restore_shape
        )

    if set(sources) == {"array"}:
        sources = []

    return predictions_output, sources
