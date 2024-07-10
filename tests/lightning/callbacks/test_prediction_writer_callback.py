"""Test PredictionWriterCallback class."""

import os
from pathlib import Path
from typing import Union
from unittest.mock import Mock, patch

import pytest
from pytorch_lightning import LightningModule, Trainer

from careamics.lightning.callbacks import PredictionWriterCallback
from careamics.lightning.callbacks.prediction_writer_callback import WriteStrategy


@pytest.fixture
def write_strategy():
    return Mock(spec=WriteStrategy)


@pytest.fixture
def dirpath(tmp_path: Path):
    return tmp_path / "predictions"


@pytest.fixture
def prediction_writer_callback(
    write_strategy: WriteStrategy, dirpath: Union[Path, str]
):
    return PredictionWriterCallback(write_strategy=write_strategy, dirpath=dirpath)


def test_initialization(prediction_writer_callback, write_strategy, dirpath):
    assert prediction_writer_callback.write_predictions is True
    assert prediction_writer_callback.write_strategy is write_strategy
    assert prediction_writer_callback.dirpath == Path(dirpath).resolve()


def test_init_dirpath_absolute_path(prediction_writer_callback):
    absolute_path = Path("/absolute/path")
    prediction_writer_callback._init_dirpath(absolute_path)
    assert prediction_writer_callback.dirpath == absolute_path


def test_init_dirpath_relative_path(prediction_writer_callback):
    relative_path = "relative/path"
    with patch("pathlib.Path.cwd", return_value=Path("/current/working/dir")):
        prediction_writer_callback._init_dirpath(relative_path)
        assert prediction_writer_callback.dirpath == Path(
            "/current/working/dir/relative/path"
        )


def test_prediction_directory_creation(prediction_writer_callback, dirpath):

    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    stage = "predict"

    prediction_writer_callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    assert os.path.isdir(dirpath)


def test_write_on_batch_end(prediction_writer_callback):
    trainer = Mock(spec=Trainer)
    pl_module = Mock(spec=LightningModule)
    prediction = Mock()
    batch_indices = Mock()
    batch = Mock()
    batch_idx = 0
    dataloader_idx = 0

    prediction_writer_callback.write_on_batch_end(
        trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    )
    prediction_writer_callback.write_strategy.write_batch.assert_called_once_with(
        trainer=trainer,
        pl_module=pl_module,
        prediction=prediction,
        batch_indices=batch_indices,
        batch=batch,
        batch_idx=batch_idx,
        dataloader_idx=dataloader_idx,
    )


# TODO: test_from_write_func_params:
#   - Only need to test that `create_write_strategy` is called
#   - & the output is `PredictionWriterCallback` with attribute write_strategy
#   - `create_write_strategy` is tested elsewhere

# def save_numpy(file_path: Path, img: NDArray, *args, **kwargs) -> None:
#     """Example custom save function."""
#     np.save(file_path, img, *args, **kwargs)


# def test_init_tiff():
#     """Test PredictionWriterCallback initialization with `save_type=="tiff"`"""
#     pwc = PredictionWriterCallback(write_type="tiff")
#     assert pwc.write_func is write_tiff
#     assert pwc.write_extension == ".tiff"


# @pytest.mark.parametrize("save_func", [None, save_numpy])
# @pytest.mark.parametrize("save_extension", [None, ".npy"])
# def test_init_custom(save_func, save_extension):
#     """Test PredictionWriterCallback initialization with `save_type=="custom"`"""
#     # Test ValueError raised with save_func or save_extension = None
#     if (save_func is None) or (save_extension is None):
#         with pytest.raises(ValueError):
#             pwc = PredictionWriterCallback(
#                 write_type="custom", write_func=save_func, write_extension=save_extension
#             )
#         return
#     pwc = PredictionWriterCallback(
#         write_type="custom", write_func=save_func, write_extension=save_extension
#     )
#     assert pwc.write_func is save_numpy
#     assert pwc.write_extension == ".npy"
