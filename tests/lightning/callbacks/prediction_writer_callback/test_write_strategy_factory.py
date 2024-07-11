"""Test write strategy factory module."""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.file_io.write import WriteFunc, write_tiff
from careamics.lightning.callbacks.prediction_writer_callback import (
    CacheTiles,
    WriteImage,
    create_write_strategy,
)
from careamics.lightning.callbacks.prediction_writer_callback.write_strategy_factory import select_write_func, select_write_extension


def save_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
    """Example custom save function."""
    np.save(file_path, img, *args, **kwargs)


def test_create_write_strategy_tiff_tiled():
    write_strategy = create_write_strategy(write_type="tiff", tiled=True)

    assert isinstance(write_strategy, CacheTiles)
    assert write_strategy.write_func is write_tiff
    assert write_strategy.write_extension == ".tiff"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_tiff_untiled():
    write_strategy = create_write_strategy(write_type="tiff", tiled=False)

    assert isinstance(write_strategy, WriteImage)
    assert write_strategy.write_func is write_tiff
    assert write_strategy.write_extension == ".tiff"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_custom_tiled():

    write_strategy = create_write_strategy(
        write_type="custom",
        tiled=True,
        write_func=save_numpy,
        write_extension=".npy"
    )
    assert isinstance(write_strategy, CacheTiles)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}


def test_create_write_strategy_custom_untiled(
):
    write_strategy = create_write_strategy(
        write_type="custom",
        tiled=False,
        write_func=save_numpy,
        write_extension=".npy"
    )
    assert isinstance(write_strategy, WriteImage)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}

def test_select_write_func_tiff():
    write_func = select_write_func(write_type="tiff", write_func=None)
    assert write_func is write_tiff


@pytest.mark.parametrize("write_func", [None, save_numpy])
def test_select_write_func_custom(write_func):
    if write_func is None:
        with pytest.raises(ValueError):
            write_func = select_write_func("custom", write_func)
        return
    write_func = select_write_func("custom", write_func)
    assert write_func is save_numpy

def test_select_write_extension_tiff():
    write_extension = select_write_extension(write_type="tiff", write_extension=None)
    assert write_extension == ".tiff"


@pytest.mark.parametrize("write_extension", [None, ".npy"])
def test_select_write_extension_custom(write_extension):
    if write_extension is None:
        with pytest.raises(ValueError):
            write_extension = select_write_extension("custom", write_extension)
        return
    write_extension = select_write_extension("custom", write_extension)
    assert write_extension == ".npy"

