"""Test write strategy factory module."""

import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray
import pytest
from unittest.mock import Mock, patch
from pytorch_lightning import Trainer, LightningModule

from careamics.file_io.write import write_tiff, WriteFunc
from careamics.lightning.callbacks import PredictionWriterCallback
from careamics.lightning.callbacks.prediction_writer_callback import (
    WriteStrategy,
    CacheTiles,
    WriteImage,
    create_write_strategy,
)


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

@pytest.mark.parametrize("write_func", [None, save_numpy])
@pytest.mark.parametrize("write_extension", [None, ".npy"])
def test_create_write_strategy_custom_tiled(
    write_func: Optional[WriteFunc], write_extension: Optional[str]
):
    if (write_func is None) or (write_extension is None):
        with pytest.raises(ValueError):
            write_strategy = create_write_strategy(
                write_type="custom",
                tiled=True,
                write_func=write_func,
                write_extension=write_extension,
            )
        return
    
    write_strategy = create_write_strategy(
        write_type="custom",
        tiled=True,
        write_func=write_func,
        write_extension=write_extension,
    )
    assert isinstance(write_strategy, CacheTiles)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}

@pytest.mark.parametrize("write_func", [None, save_numpy])
@pytest.mark.parametrize("write_extension", [None, ".npy"])
def test_create_write_strategy_custom_untiled(
    write_func: Optional[WriteFunc], write_extension: Optional[str]
):
    if (write_func is None) or (write_extension is None):
        with pytest.raises(ValueError):
            write_strategy = create_write_strategy(
                write_type="custom",
                tiled=True,
                write_func=write_func,
                write_extension=write_extension,
            )
        return
    
    write_strategy = create_write_strategy(
        write_type="custom",
        tiled=False,
        write_func=write_func,
        write_extension=write_extension,
    )
    assert isinstance(write_strategy, WriteImage)
    assert write_strategy.write_func is save_numpy
    assert write_strategy.write_extension == ".npy"
    assert write_strategy.write_func_kwargs == {}


# TODO: test_select_write_func maybe move some asserts and raises from above tests.
# TODO: test_select_write_extension ^similar