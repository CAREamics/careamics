"""Module containing file path utilities for `WriteStrategy` to use."""

from pathlib import Path
from typing import Union

from careamics.compat.dataset import IterablePredDataset, IterableTiledPredDataset


# TODO: move to datasets package ?
def get_sample_file_path(
    dataset: Union[IterableTiledPredDataset, IterablePredDataset], sample_id: int
) -> Path:
    """
    Get the file path for a particular sample.

    Parameters
    ----------
    dataset : IterableTiledPredDataset or IterablePredDataset
        Dataset.
    sample_id : int
        Sample ID, the index of the file in the dataset `dataset`.

    Returns
    -------
    Path
        The file path corresponding to the sample with the ID `sample_id`.
    """
    return dataset.data_files[sample_id]
