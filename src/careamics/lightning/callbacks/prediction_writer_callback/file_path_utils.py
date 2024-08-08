"""Module containing file path utilities for `WriteStrategy` to use."""

from pathlib import Path
from typing import Union

import numpy as np

from careamics.dataset import IterablePredDataset, IterableTiledPredDataset


# TODO: move to datasets package ?
def get_sample_file_path(
    dataset: Union[IterableTiledPredDataset, IterablePredDataset], sample_index: int
) -> Path:
    """
    Get the file path for a particular sample.

    Parameters
    ----------
    dataset : IterableTiledPredDataset or IterablePredDataset
        Dataset.
    sample_index : int
        The sample index, (in the order samples are iterated over in
        `IterableTiledPredDataset` or `IterablePredDataset`.

    Returns
    -------
    Path
        The file path corresponding to the sample with the ID `sample_id`.
    """
    if len(dataset.sample_file_indices) == 0:
        raise ValueError("No samples found. Dataset has likely not started iterating.")
    file_index: int = dataset.sample_file_indices[sample_index]
    file_path: Path = dataset.data_files[file_index]
    if "S" in dataset.axes:
        sample_start = np.where(np.array(dataset.sample_file_indices) == file_index)[0][
            0
        ]
        sample_id = sample_index - sample_start

        sample_file_name = f"{file_path.stem}_{sample_id}"
        file_path = (file_path.parent / sample_file_name).with_suffix(file_path.suffix)

    return file_path


def create_write_file_path(
    dirpath: Path, file_path: Path, write_extension: str
) -> Path:
    """
    Create the file name for the output file.

    Takes the original file path, changes the directory to `dirpath` and changes
    the extension to `write_extension`.

    Parameters
    ----------
    dirpath : pathlib.Path
        The output directory to write file to.
    file_path : pathlib.Path
        The original file path.
    write_extension : str
        The extension that output files should have.

    Returns
    -------
    Path
        The output file path.
    """
    file_name = Path(file_path.stem).with_suffix(write_extension)
    file_path = dirpath / file_name
    return file_path
