import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import tifffile

from careamics.config import Configuration
from careamics.config.algorithm import Algorithm
from careamics.config.data import Data
from careamics.config.training import LrScheduler, Optimizer, Training
from careamics.engine import Engine


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def example_data_path(temp_dir: Path) -> Tuple[Path, Path]:
    def _example_data_path(image_size: Tuple[int, int]):
        test_image = np.random.rand(*image_size)
        test_image_predict = test_image[None, None, ...]

        train_path = temp_dir / "train"
        val_path = temp_dir / "val"
        test_path = temp_dir / "test"
        train_path.mkdir()
        val_path.mkdir()
        test_path.mkdir()

        tifffile.imwrite(train_path / "train_image.tif", test_image)
        tifffile.imwrite(val_path / "val_image.tif", test_image)
        tifffile.imwrite(test_path / "test_image.tif", test_image_predict)

        return train_path, val_path, test_path

    return _example_data_path


@pytest.fixture
def base_configuration(temp_dir: Path, patch_size: Tuple[int, int]) -> Configuration:
    def _base_configuration(axes: str) -> Configuration:
        is_3d = "Z" in axes
        configuration = Configuration(
            experiment_name="smoke_test",
            working_directory=temp_dir,
            algorithm=Algorithm(loss="n2v", model="UNet", is_3D=str(is_3d)),
            data=Data(
                in_memory=True,
                data_format="tif",
                axes=axes,
            ),
            training=Training(
                num_epochs=1,
                patch_size=patch_size,
                batch_size=1,
                optimizer=Optimizer(name="Adam"),
                lr_scheduler=LrScheduler(name="ReduceLROnPlateau"),
                extraction_strategy="random",
                augmentation=True,
                num_workers=0,
                use_wandb=False,
            ),
        )
        return configuration

    return _base_configuration


@pytest.mark.parametrize(
    "image_size, axes, patch_size, overlaps",
    [
        ((64, 64), "YX", (32, 32), (8, 8)),
        ((2, 64, 64), "SYX", (32, 32), (8, 8)),
        # ((16, 64, 64), "ZYX", (8, 32, 32), (2, 8, 8)),
        # ((2, 16, 64, 64), "SZYX", (8, 32, 32), (2, 8, 8)),
    ],
)
def test_is_engine_runnable(
    base_configuration: Configuration,
    example_data_path: Tuple[Path, Path],
    image_size: Tuple[int, int],
    axes: str,
    patch_size: Tuple[int, int],
    overlaps: Tuple[int, int],
):
    """
    Test if basic workflow does not fail - train model and then predict
    """
    train_path, val_path, test_path = example_data_path(image_size)
    configuration = base_configuration(axes)
    engine = Engine(config=configuration)
    _ = engine.train(train_path, val_path)

    model_name = f"{engine.cfg.experiment_name}_best.pth"
    result_model_path = engine.cfg.working_directory / model_name

    assert result_model_path.exists()

    # Test prediction with external input
    test_image = np.random.rand(*image_size)
    test_result = engine.predict(input=test_image)

    assert test_result is not None

    # Test prediction with pred_path without tiling
    test_result = engine.predict(input=test_path)

    assert test_result is not None

    # save as bioimage
    zip_path = Path(configuration.working_directory) / "model.bioimage.io.zip"
    engine.save_as_bioimage(zip_path)
    assert zip_path.exists()

    # Create engine from checkpoint
    second_engine = Engine(model_path=result_model_path)
    second_engine.cfg.data.in_memory = False
    _ = second_engine.train(train_path, val_path)

    # Create engine from bioimage model
    third_engine = Engine(model_path=zip_path)
    third_engine.cfg.data.in_memory = False
    _ = third_engine.train(train_path, val_path)

    # Test prediction with pred_path with tiling
    test_result = third_engine.predict(
        input=test_path, tile_shape=patch_size, overlaps=overlaps
    )
    assert test_result is not None
