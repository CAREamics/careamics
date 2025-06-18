"""Module containing functions to convert prediction outputs to desired form."""

from typing import Any, Literal, Union, overload

import numpy as np
from numpy.typing import NDArray

from ..config.tile_information import TileInformation
from .stitch_prediction import stitch_prediction


def convert_outputs(predictions: list[Any], tiled: bool) -> list[NDArray]:
    """
    Convert the Lightning trainer outputs to the desired form.

    This method allows stitching back together tiled predictions.

    Parameters
    ----------
    predictions : list
        Predictions that are output from `Trainer.predict`.
    tiled : bool
        Whether the predictions are tiled.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        list of arrays with the axes SC(Z)YX. If there is only 1 output it will not
        be in a list.
    """
    if len(predictions) == 0:
        return predictions

    # this layout is to stop mypy complaining
    if tiled:
        predictions_comb = combine_batches(predictions, tiled)
        predictions_output = stitch_prediction(*predictions_comb)
    else:
        predictions_output = combine_batches(predictions, tiled)

    return predictions_output


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: list[Any], tiled: Literal[True]
) -> tuple[list[NDArray], list[TileInformation]]: ...


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: list[Any], tiled: Literal[False]
) -> list[NDArray]: ...


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: list[Any], tiled: Union[bool, Literal[True], Literal[False]]
) -> Union[list[NDArray], tuple[list[NDArray], list[TileInformation]]]: ...


def combine_batches(
    predictions: list[Any], tiled: bool
) -> Union[list[NDArray], tuple[list[NDArray], list[TileInformation]]]:
    """
    If predictions are in batches, they will be combined.

    Parameters
    ----------
    predictions : list
        Predictions that are output from `Trainer.predict`.
    tiled : bool
        Whether the predictions are tiled.

    Returns
    -------
    (list of numpy.ndarray) or tuple of (list of numpy.ndarray, list of TileInformation)
        Combined batches.
    """
    if tiled:
        return _combine_tiled_batches(predictions)
    else:
        return _combine_array_batches(predictions)


def _combine_tiled_batches(
    predictions: list[tuple[NDArray, list[TileInformation]]],
) -> tuple[list[NDArray], list[TileInformation]]:
    """
    Combine batches from tiled output.

    Parameters
    ----------
    predictions : list of (numpy.ndarray, list of TileInformation)
        Predictions that are output from `Trainer.predict`. For tiled batches, this is
        a list of tuples. The first element of the tuples is the prediction output of
        tiles with dimension (B, C, (Z), Y, X), where B is batch size. The second
        element of the tuples is a list of TileInformation objects of length B.

    Returns
    -------
    tuple of (list of numpy.ndarray, list of TileInformation)
        Combined batches.
    """
    # turn list of lists into single list
    tile_infos = [
        tile_info for _, tile_info_list in predictions for tile_info in tile_info_list
    ]
    prediction_tiles: list[NDArray] = _combine_array_batches(
        [preds for preds, _ in predictions]
    )
    return prediction_tiles, tile_infos


def _combine_array_batches(predictions: list[NDArray]) -> list[NDArray]:
    """
    Combine batches of arrays.

    Parameters
    ----------
    predictions : list
        Prediction arrays that are output from `Trainer.predict`. A list of arrays that
        have dimensions (B, C, (Z), Y, X), where B is batch size.

    Returns
    -------
    list of numpy.ndarray
        A list of arrays with dimensions (1, C, (Z), Y, X).
    """
    prediction_concat: NDArray = np.concatenate(predictions, axis=0)
    prediction_split = np.split(prediction_concat, prediction_concat.shape[0], axis=0)
    return prediction_split
