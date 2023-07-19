import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import yaml
from tifffile import tifffile

from careamics_restoration.config import Configuration
from careamics_restoration.config.algorithm import Algorithm
from careamics_restoration.config.data import Data
from careamics_restoration.config.prediction import Prediction
from careamics_restoration.config.training import LrScheduler, Optimizer, Training
from careamics_restoration.engine import Engine

TEST_IMAGE_SIZE = (128, 128)
TEST_PATCH_SIZE = (64, 64)


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def example_data_path(temp_dir: Path) -> Tuple[Path, Path]:
    test_image = np.random.rand(*TEST_IMAGE_SIZE)

    train_path = temp_dir / "train"
    val_path = temp_dir / "val"
    train_path.mkdir()
    val_path.mkdir()

    tifffile.imwrite(train_path / "train_image.tif", test_image)
    tifffile.imwrite(val_path / "val_image.tif", test_image)

    return train_path, val_path


@pytest.fixture
def base_configuration(
    temp_dir: Path, example_data_path: Tuple[Path, Path]
) -> Configuration:
    train_dir, val_dir = example_data_path
    configuration = Configuration(
        experiment_name="smoke_test",
        working_directory=temp_dir,
        algorithm=Algorithm(loss="n2v", model="UNet", is_3D="False"),
        data=Data(
            data_format="tif",
            axes="YX",
            training_path=train_dir,
            validation_path=val_dir,
            prediction_path=val_dir,
        ),
        training=Training(
            num_epochs=1,
            patch_size=TEST_PATCH_SIZE,
            batch_size=1,
            optimizer=Optimizer(name="Adam"),
            lr_scheduler=LrScheduler(name="ReduceLROnPlateau"),
            extraction_strategy="random",
            augmentation=True,
            num_workers=0,
            use_wandb=False,
        ),
        prediction=Prediction(use_tiling=False),
    )
    return configuration


def dump_config(configuration: Configuration) -> Path:
    temp_dir = configuration.working_directory
    config_path = temp_dir / "test_config.yml"
    config_dict = configuration.model_dump()
    with open(config_path, "w") as config_file:
        yaml.dump(config_dict, config_file)
    return config_path


def test_is_engine_runnable(base_configuration: Configuration):
    """
    Test if basic workflow does not fail - train model and then predict
    """
    engine = Engine(config=base_configuration)
    engine.train()

    model_name = f"{engine.cfg.experiment_name}_best.pth"
    result_model_path = engine.cfg.working_directory / model_name

    assert result_model_path.exists()

    test_image = np.random.rand(*TEST_IMAGE_SIZE)

    # Predict only accepts 4D input for now
    test_image = test_image[None, None, ...]
    test_result = engine.predict(external_input=test_image, mean=0.5, std=0.5)

    assert test_result is not None
