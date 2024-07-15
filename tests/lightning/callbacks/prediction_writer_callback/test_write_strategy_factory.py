"""Test write strategy factory module."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.file_io.write import write_tiff
from careamics.lightning.callbacks.prediction_writer_callback import (
    CacheTiles,
    WriteImage,
    create_write_strategy,
    select_write_extension,
    select_write_func,
)


def save_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
    """Example custom save function."""
    np.save(file_path, img, *args, **kwargs)


def test_create_write_strategy_tiff_tiled():
    """Test write strategy creation for tiled tiff."""
    write_strategy = create_write_strategy(write_type="tiff", tiled=True)

    assert isinstance(write_strategy, CacheTiles)
    assert write_strategy.write_func is write_tiff
    assert write_strategy.write_extension == ".tiff"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_tiff_untiled():
    """Test write strategy creation for untiled tiff."""
    write_strategy = create_write_strategy(write_type="tiff", tiled=False)

    assert isinstance(write_strategy, WriteImage)
    assert write_strategy.write_func is write_tiff
    assert write_strategy.write_extension == ".tiff"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_custom_tiled():
    """Test write strategy creation for tiled custom type."""
    write_strategy = create_write_strategy(
        write_type="custom", tiled=True, write_func=save_numpy, write_extension=".npy"
    )
    assert isinstance(write_strategy, CacheTiles)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_custom_untiled():
    """Test write strategy creation for untiled custom type."""
    write_strategy = create_write_strategy(
        write_type="custom", tiled=False, write_func=save_numpy, write_extension=".npy"
    )
    assert isinstance(write_strategy, WriteImage)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}


def test_select_write_func_tiff():
    """Test tiff write function is selected correctly."""
    write_func = select_write_func(write_type="tiff", write_func=None)
    assert write_func is write_tiff


@pytest.mark.parametrize("write_func", [None, save_numpy])
def test_select_write_func_custom(write_func):
    """Test custom write func is returned or raises error if None."""
    if write_func is None:
        with pytest.raises(ValueError):
            write_func = select_write_func("custom", write_func)
        return
    write_func = select_write_func("custom", write_func)
    assert write_func is save_numpy


def test_select_write_extension_tiff():
    """Test tiff extension is selected correctly."""
    write_extension = select_write_extension(write_type="tiff", write_extension=None)
    assert write_extension == ".tiff"


@pytest.mark.parametrize("write_extension", [None, ".npy"])
def test_select_write_extension_custom(write_extension):
    """Test custom extension is returned or raises if None."""
    if write_extension is None:
        with pytest.raises(ValueError):
            write_extension = select_write_extension("custom", write_extension)
        return
    write_extension = select_write_extension("custom", write_extension)
    assert write_extension == ".npy"
