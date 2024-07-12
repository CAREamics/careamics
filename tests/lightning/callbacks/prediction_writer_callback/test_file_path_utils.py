from pathlib import Path
from unittest.mock import Mock

import pytest

from careamics.config import InferenceConfig
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.lightning.callbacks.prediction_writer_callback.file_path_utils import (
    create_write_file_path,
    get_sample_file_path,
)


def iterable_pred_ds():
    """`IterablePredDataset` with mock prediction config."""
    src_files = [f"{i}.ext" for i in range(2)]
    pred_config = Mock(spec=InferenceConfig)
    # attrs used in DS initialization
    pred_config.axes = Mock()
    pred_config.image_means = [Mock()]
    pred_config.image_stds = [Mock()]
    ds = IterablePredDataset(pred_config, src_files=src_files)
    return ds


def iterable_tiled_pred_ds():
    """`IterableTiledPredDataset` with mock prediction config."""
    src_files = [f"{i}.ext" for i in range(2)]
    pred_config = Mock(spec=InferenceConfig)
    # attrs used in DS initialization
    pred_config.axes = Mock()
    pred_config.tile_size = Mock()
    pred_config.tile_overlap = Mock()
    pred_config.image_means = [Mock()]
    pred_config.image_stds = [Mock()]
    ds = IterableTiledPredDataset(pred_config, src_files=src_files)
    return ds


@pytest.mark.parametrize("ds_func", [iterable_pred_ds, iterable_tiled_pred_ds])
def test_get_sample_file_path(ds_func):
    ds = ds_func()
    for i in range(2):
        file_path = get_sample_file_path(ds, sample_id=i)
        assert file_path == f"{i}.ext"


def test_create_write_file_path():
    dirpath = Path("output_directory")
    file_path = Path("input_directory/file_name.in_ext")
    write_extension = ".out_ext"

    write_file_path = create_write_file_path(
        dirpath=dirpath, file_path=file_path, write_extension=write_extension
    )
    assert write_file_path == Path("output_directory/file_name.out_ext")
