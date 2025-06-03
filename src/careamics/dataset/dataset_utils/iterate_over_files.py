"""Function to iterate over files."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Optional, Union

from numpy.typing import NDArray
from torch.utils.data import get_worker_info

from careamics.config import DataConfig, InferenceConfig
from careamics.file_io.read import read_tiff
from careamics.utils.logging import get_logger

from .dataset_utils import reshape_array

logger = get_logger(__name__)


def iterate_over_files(
    data_config: Union[DataConfig, InferenceConfig],
    data_files: list[Path],
    target_files: Optional[list[Path]] = None,
    read_source_func: Callable = read_tiff,
) -> Generator[tuple[NDArray, Optional[NDArray]], None, None]:
    """Iterate over data source and yield whole reshaped images.

    Parameters
    ----------
    data_config : CAREamics DataConfig or InferenceConfig
        Configuration.
    data_files : list of pathlib.Path
        List of data files.
    target_files : list of pathlib.Path, optional
        List of target files, by default None.
    read_source_func : Callable, optional
        Function to read the source, by default read_tiff.

    Yields
    ------
    NDArray
        Image.
    """
    # When num_workers > 0, each worker process will have a different copy of the
    # dataset object
    # Configuring each copy independently to avoid having duplicate data returned
    # from the workers
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else 0
    num_workers = worker_info.num_workers if worker_info is not None else 1

    # iterate over the files
    for i, filename in enumerate(data_files):
        # retrieve file corresponding to the worker id
        if i % num_workers == worker_id:
            try:
                # read data
                sample = read_source_func(filename, data_config.axes)

                # reshape array
                reshaped_sample = reshape_array(sample, data_config.axes)

                # read target, if available
                if target_files is not None:
                    if filename.name != target_files[i].name:
                        raise ValueError(
                            f"File {filename} does not match target file "
                            f"{target_files[i]}. Have you passed sorted "
                            f"arrays?"
                        )

                    # read target
                    target = read_source_func(target_files[i], data_config.axes)

                    # reshape target
                    reshaped_target = reshape_array(target, data_config.axes)

                    yield reshaped_sample, reshaped_target
                else:
                    yield reshaped_sample, None

            except Exception as e:
                logger.error(f"Error reading file {filename}: {e}")
