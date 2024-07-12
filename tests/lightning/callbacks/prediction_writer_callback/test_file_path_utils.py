from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from careamics.config import InferenceConfig
from careamics.lightning.callbacks.prediction_writer_callback.file_path_utils import (
    get_sample_file_path,
    create_write_file_path,
)
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset


@pytest.fixture
def iterable_pred_ds():
    """`IterablePredDataset` with mock prediction config."""
    src_files = [f"{i}.ext" for i in range(2)]
    ds = IterablePredDataset(Mock(spec=InferenceConfig), src_files=src_files)
    return ds


@pytest.fixture
def iterable_tiled_pred_ds():
    """`IterableTiledPredDataset` with mock prediction config."""
    src_files = [f"{i}.ext" for i in range(2)]
    ds = IterableTiledPredDataset(Mock(spec=InferenceConfig), src_files=src_files)
    return ds


@pytest.mark.parametrize("ds", [iterable_pred_ds, iterable_tiled_pred_ds])
def test_get_sample_file_path(ds):
    for i in range(2):
        file_path = get_sample_file_path(ds)
        assert file_path == f"{i}.ext"


def test_create_write_file_path():
    dirpath = Path("output_directory")
    file_path = Path("input_directory/file_name.in_ext")
    write_extension = ".out_ext"

    write_file_path = create_write_file_path(
        dirpath=dirpath, file_path=file_path, write_extension=write_extension
    )
    assert write_file_path == Path("output_directory/file_name.out_ext")
