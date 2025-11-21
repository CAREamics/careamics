"""Module containing functions to convert prediction outputs to desired form."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData

from .stitch_prediction import stitch_prediction


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
    - chunks: either a single tuple (1,) or a sequence of tuples of length B

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
        region_spec = {
            key: (
                tuple(int(value[idx][i]) for idx in range(len(value)))
                if isinstance(value, list)
                else int(value[i])
            )  # handles tensor (1D) vs list of tensors/tuples (2D)
            for key, value in batch.region_spec.items()
        }

        # handle chunks being either a single tuple or a sequence of tuples
        if isinstance(batch.chunks, list):
            chunks: Sequence[int] = tuple(int(val[i]) for val in batch.chunks)
        else:
            chunks = batch.chunks

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
            chunks=chunks,
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
    predictions_by_data_idx: defaultdict[int, list[ImageRegionData]] = defaultdict(list)
    for image_region in predictions:
        data_idx = image_region.region_spec["data_idx"]
        predictions_by_data_idx[data_idx].append(image_region)

    # sort predictions by sample idx
    combined_predictions: list[NDArray] = []
    combined_sources: list[str] = []
    for data_idx in sorted(predictions_by_data_idx.keys()):
        image_regions = predictions_by_data_idx[data_idx]
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
) -> list[NDArray]:
    """
    Convert the Lightning trainer outputs to the desired form.

    This method allows decollating batches and stitching back together tiled
    predictions.

    Parameters
    ----------
    predictions : list[ImageRegionData]
        Output from `Trainer.predict`, list of batches.
    tiled : bool
        Whether the predictions are tiled.

    Returns
    -------
    list of numpy.ndarray
        list of arrays with the axes SC(Z)YX. If there is only 1 output it will not
        be in a list.
    """
    # decollate batches
    decollated_predictions: list[ImageRegionData] = []
    for batch in predictions:
        decollated_batch = decollate_image_region_data(batch)
        decollated_predictions.extend(decollated_batch)

    if not tiled and "tot_tiles" in decollated_predictions[0].region_spec:
        raise ValueError(
            "Predictions contain 'tot_tiles' in region_spec but tiled is set to False."
        )

    if tiled:
        predictions_output = stitch_prediction(decollated_predictions)
    else:
        predictions_output, _ = combine_samples(decollated_predictions)
    # TODO squeeze single output?
    return predictions_output
