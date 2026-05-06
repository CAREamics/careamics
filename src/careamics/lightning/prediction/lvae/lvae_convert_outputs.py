"""Convert VAE outputs."""

import numpy as np
from numpy.typing import NDArray

from .lvae_stitch_prediction import stitch_prediction_vae


def convert_outputs_microsplit(
    predictions: list[tuple[NDArray, NDArray]], dataset
) -> tuple[NDArray, NDArray]:
    """
    Convert microsplit Lightning trainer outputs using eval_utils stitching functions.

    This function processes microsplit predictions that return
    (tile_prediction, tile_std) tuples and stitches them back together using the same
    logic as get_single_file_mmse.

    Parameters
    ----------
    predictions : list of tuple[NDArray, NDArray]
        Predictions from Lightning trainer for microsplit. Each element is a tuple of
        (tile_prediction, tile_std) where both are numpy arrays from predict_step.
    dataset : Dataset
        The dataset object used for prediction, needed for stitching function selection
        and stitching process.

    Returns
    -------
    tuple[NDArray, NDArray]
        A tuple of (stitched_predictions, stitched_stds) representing the full
        stitched predictions and standard deviations.
    """
    if len(predictions) == 0:
        raise ValueError("No predictions provided")

    # Separate predictions and stds from the list of tuples
    tile_predictions = [pred for pred, _ in predictions]
    tile_stds = [std for _, std in predictions]

    # Concatenate all tiles exactly like get_single_file_mmse
    tiles_arr = np.concatenate(tile_predictions, axis=0)
    tile_stds_arr = np.concatenate(tile_stds, axis=0)

    # Apply stitching using stitch_predictions_new
    stitched_predictions = stitch_prediction_vae(tiles_arr, dataset)
    stitched_stds = stitch_prediction_vae(tile_stds_arr, dataset)

    return stitched_predictions, stitched_stds
