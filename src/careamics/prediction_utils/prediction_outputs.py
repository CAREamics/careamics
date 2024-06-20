"""Module containing functions to convert prediction outputs to desired form."""

from typing import Any, List, Literal, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray

from ..config.tile_information import TileInformation
from .stitch_prediction import stitch_prediction


def convert_outputs(
    predictions: List[Any], tiled: bool
) -> Union[List[NDArray], NDArray]:
    """
    Convert the outputs to the desired form.

    Parameters
    ----------
    predictions : list
        Predictions that are output from `Trainer.predict`.
    tiled : bool
        Whether the predictions are tiled.

    Returns
    -------
    list of numpy.ndarray or numpy.ndarray
        List of arrays with the axes SC(Z)YX. If there is only 1 output it will not
        be in a list.
    """
    if len(predictions) == 0:
        return predictions

    # this layout is to stop mypy complaining
    if tiled:
        predictions_comb = combine_batches(predictions, tiled)
        # remove sample dimension (always 1) `stitch_predict` func expects no S dim
        tiles = [pred[0] for pred in predictions_comb[0]]
        tile_infos = predictions_comb[1]
        predictions_output = stitch_prediction(tiles, tile_infos)
    else:
        predictions_output = combine_batches(predictions, tiled)

    # TODO: add this in? Returns output with same axes as input
    # Won't work with tiling rn because stitch_prediction func removes S axis
    # predictions = reshape(predictions, axes)
    # At least make sure stitched prediction and non-tiled prediction have matching axes

    # TODO: might want to remove this
    if len(predictions_output) == 1:
        return predictions_output[0]
    return predictions_output


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: List[Any], tiled: Literal[True]
) -> Tuple[List[NDArray], List[TileInformation]]: ...


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: List[Any], tiled: Literal[False]
) -> List[NDArray]: ...


# for mypy
@overload
def combine_batches(  # numpydoc ignore=GL08
    predictions: List[Any], tiled: Union[bool, Literal[True], Literal[False]]
) -> Union[List[NDArray], Tuple[List[NDArray], List[TileInformation]]]: ...


def combine_batches(
    predictions: List[Any], tiled: bool
) -> Union[List[NDArray], Tuple[List[NDArray], List[TileInformation]]]:
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
        return _combine_untiled_batches(predictions)


def _combine_tiled_batches(
    predictions: List[Tuple[NDArray, List[TileInformation]]]
) -> Tuple[List[NDArray], List[TileInformation]]:
    """
    Combine batches from tiled output.

    Parameters
    ----------
    predictions : list
        Predictions that are output from `Trainer.predict`.

    Returns
    -------
    tuple of (list of numpy.ndarray, list of TileInformation)
        Combined batches.
    """
    # turn list of lists into single list
    tile_infos = [
        tile_info for _, tile_info_list in predictions for tile_info in tile_info_list
    ]
    prediction_tiles: List[NDArray] = _combine_untiled_batches(
        [preds for preds, _ in predictions]
    )
    return prediction_tiles, tile_infos


def _combine_untiled_batches(predictions: List[NDArray]) -> List[NDArray]:
    """
    Combine batches from un-tiled output.

    Parameters
    ----------
        predictions : list
            Predictions that are output from `Trainer.predict`.

    Returns
    -------
        list of nunpy.ndarray
            Combined batches.
    """
    prediction_concat: NDArray = np.concatenate(predictions, axis=0)
    prediction_split = np.split(prediction_concat, prediction_concat.shape[0], axis=0)
    return prediction_split


def reshape(predictions: List[NDArray], axes: str) -> List[NDArray]:
    """
    Reshape predictions to have dimensions of input.

    Parameters
    ----------
    predictions : list
        Predictions that are output from `Trainer.predict`.
    axes : str
        Axes SC(Z)YX.

    Returns
    -------
    List[NDArray]
        Reshaped predicitions.
    """
    if "C" not in axes:
        predictions = [pred[:, 0] for pred in predictions]
    if "S" not in axes:
        predictions = [pred[0] for pred in predictions]
    return predictions
