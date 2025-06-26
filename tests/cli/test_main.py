from pathlib import Path

import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from careamics import CAREamist, Configuration
from careamics.cli.main import app
from careamics.config import save_configuration
from careamics.config.support import SupportedData

pytestmark = pytest.mark.mps_gh_fail

runner = CliRunner()


def test_train(tmp_path: Path, minimum_n2v_configuration: dict):

    # create & save config
    config_path = tmp_path / "config.yaml"
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.data_type = SupportedData.TIFF.value
    save_configuration(config, config_path)

    # training data
    train_array = np.random.rand(32, 32)
    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    # invoke command
    result = runner.invoke(
        app,
        [
            "train",
            str(config_path),
            "-ts",
            str(train_file),
            "-wd",
            str(tmp_path),
        ],
    )
    assert (tmp_path / "checkpoints").is_dir()
    assert len(list((tmp_path / "checkpoints").glob("*.ckpt"))) > 0
    assert result.exit_code == 0


def test_predict_single_file(tmp_path: Path, minimum_n2v_configuration: dict):

    # create & save config
    config_path = tmp_path / "config.yaml"
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.data_type = SupportedData.TIFF.value
    save_configuration(config, config_path)

    # dummy data
    train_array = np.random.rand(32, 32)
    # save files
    train_file = tmp_path / "train.tiff"
    tifffile.imwrite(train_file, train_array)

    careamist = CAREamist(config, work_dir=tmp_path)
    careamist.train(train_source=train_file)

    checkpoint_path = next(iter((tmp_path / "checkpoints").glob("*.ckpt")))

    result = runner.invoke(
        app, ["predict", str(checkpoint_path), str(train_file), "-wd", str(tmp_path)]
    )
    assert (tmp_path / "predictions").is_dir()
    assert (tmp_path / "predictions" / "train.tiff").is_file()
    assert result.exit_code == 0


def test_predict_directory(tmp_path: Path, minimum_n2v_configuration: dict):

    # create & save config
    config_path = tmp_path / "config.yaml"
    config = Configuration(**minimum_n2v_configuration)
    config.data_config.data_type = SupportedData.TIFF.value
    save_configuration(config, config_path)

    n_files = 2
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # dummy data
    for i in range(n_files):
        train_array = np.random.rand(32, 32)
        # save files
        train_file = data_dir / f"train_{i}.tiff"
        tifffile.imwrite(train_file, train_array)

    careamist = CAREamist(config, work_dir=tmp_path)
    careamist.train(train_source=data_dir)

    checkpoint_path = next(iter((tmp_path / "checkpoints").glob("*.ckpt")))

    result = runner.invoke(
        app, ["predict", str(checkpoint_path), str(data_dir), "-wd", str(tmp_path)]
    )
    assert (tmp_path / "predictions").is_dir()
    for i in range(n_files):
        assert (tmp_path / "predictions" / f"train_{i}.tiff").is_file()
    assert result.exit_code == 0
