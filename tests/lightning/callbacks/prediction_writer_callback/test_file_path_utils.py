from pathlib import Path
from unittest.mock import Mock

from careamics.config import InferenceConfig
from careamics.dataset import IterablePredDataset, IterableTiledPredDataset
from careamics.lightning.callbacks.prediction_writer_callback.file_path_utils import (
    create_write_file_path,
    get_sample_file_path,
)


def test_get_sample_file_path_tiled_ds():

    # Create DS with mock InferenceConfig
    src_files = [f"{i}.ext" for i in range(2)]
    pred_config = Mock(spec=InferenceConfig)
    # attrs used in DS initialization
    pred_config.axes = Mock()
    pred_config.tile_size = Mock()
    pred_config.tile_overlap = Mock()
    pred_config.image_means = [Mock()]
    pred_config.image_stds = [Mock()]
    ds = IterableTiledPredDataset(pred_config, src_files=src_files)

    for i in range(2):
        file_path = get_sample_file_path(ds, sample_id=i)
        assert file_path == f"{i}.ext"


def test_get_sample_file_path_untiled_ds():

    # Create DS with mock InferenceConfig
    src_files = [f"{i}.ext" for i in range(2)]
    pred_config = Mock(spec=InferenceConfig)
    # attrs used in DS initialization
    pred_config.axes = Mock()
    pred_config.image_means = [Mock()]
    pred_config.image_stds = [Mock()]
    ds = IterablePredDataset(pred_config, src_files=src_files)

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
