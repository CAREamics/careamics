"""Test `WriteImage` class."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.dataset_ng.dataset import ImageRegionData
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    WriteImage,
    create_write_file_path,
)

# TODO test when multiple sample


@pytest.fixture
def write_func():
    def a_write_func(file_path: Path, img: NDArray, *args, **kwargs) -> None:
        """Do-nothing write function."""
        return

    return a_write_func


@pytest.fixture
def write_image_strategy(write_func) -> WriteImage:
    """
    Initialized `WriteImage` class.

    Parameters
    ----------
    write_func : WriteFunc
        Write function. (Comes from fixture).

    Returns
    -------
    CacheTiles
        Initialized `CacheTiles` class.
    """
    write_extension = ".ext"
    write_func_kwargs = {}
    return WriteImage(
        write_func=write_func,
        write_extension=write_extension,
        write_func_kwargs=write_func_kwargs,
    )


def test_cache_tiles_init(write_func, write_image_strategy):
    """
    Test `WriteImage` initializes as expected.
    """
    assert write_image_strategy.write_func is write_func
    assert write_image_strategy.write_extension == ".ext"
    assert write_image_strategy.write_func_kwargs == {}


def test_write_image_batch(write_image_strategy, ordered_array, mocker):
    """Test writing a batch."""
    array = ordered_array((5, 8, 8))

    prediction = [
        ImageRegionData(
            source="array.tiff",
            data=array[i : i + 1],  # keep S dim as singleton to mock C dim
            data_shape=array.shape,
            dtype=np.float32,
            axes="SYX",
            region_spec={
                "data_idx": 0,
                "sample_idx": i,
                "coords": (0, 0),
                "patch_size": (8, 8),
            },
            additional_metadata={},
        )
        for i in range(array.shape[0])
    ]

    # spy on write function
    spy = mocker.spy(write_image_strategy, "write_func")

    # call write batch
    dirpath = Path("out_dir")
    write_image_strategy.write_batch(
        dirpath=dirpath,
        predictions=prediction,
    )

    # check call and arguments
    spy.assert_called_once()

    call_args = spy.call_args.kwargs
    expected_file_path = create_write_file_path(
        dirpath=dirpath,
        file_path=prediction[0].source,
        write_extension=write_image_strategy.write_extension,
    )
    assert call_args["file_path"] == expected_file_path
    np.testing.assert_array_equal(call_args["img"], array)
