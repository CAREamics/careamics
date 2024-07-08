"""Test PredictionWriterCallback class."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from careamics.file_io.write import write_tiff
from careamics.lightning.callbacks import PredictionWriterCallback


def save_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
    """Example custom save function."""
    np.save(file_path, img, *args, **kwargs)


def test_init_tiff():
    """Test PredictionWriterCallback initialization with `save_type=="tiff"`"""
    pwc = PredictionWriterCallback(save_type="tiff")
    assert pwc.save_func is write_tiff
    assert pwc.save_extension == ".tiff"


@pytest.mark.parametrize("save_func", [None, save_numpy])
@pytest.mark.parametrize("save_extension", [None, ".npy"])
def test_init_custom(save_func, save_extension):
    """Test PredictionWriterCallback initialization with `save_type=="custom"`"""
    # Test ValueError raised with save_func or save_extension = None
    if (save_func is None) or (save_extension is None):
        with pytest.raises(ValueError):
            pwc = PredictionWriterCallback(
                save_type="custom", save_func=save_func, save_extension=save_extension
            )
        return
    pwc = PredictionWriterCallback(
        save_type="custom", save_func=save_func, save_extension=save_extension
    )
    assert pwc.save_func is save_numpy
    assert pwc.save_extension == ".npy"
