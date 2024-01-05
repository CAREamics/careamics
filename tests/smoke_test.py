from pathlib import Path
from typing import Tuple

import numpy as np

from careamics.config import Configuration
from careamics.engine import Engine


def test_is_engine_runnable(
    base_configuration: Configuration,
    supervised_configuration: Configuration,
    example_data_path: Tuple[Path, Path],
    image_size: Tuple[int, int],
    patch_size: Tuple[int, int],
    overlaps: Tuple[int, int],
):
    """
    Test if basic workflow does not fail - train model and then predict
    """
    train_path, val_path, test_path = example_data_path
    engine = Engine(config=base_configuration)
    _ = engine.train(train_path, val_path)

    model_name = f"{engine.cfg.experiment_name}_best.pth"
    result_model_path = engine.cfg.working_directory / model_name

    assert result_model_path.exists()

    # Test prediction with external input
    test_image = np.random.rand(*image_size)

    test_result = engine.predict(
        inputs=test_image, axes="YX", tile_shape=patch_size, overlaps=overlaps
    )
    assert test_result is not None

    # Test prediction with external input without tiling
    test_result = engine.predict(inputs=test_image[None, ...], axes="YX")
    assert test_result is not None

    # Test prediction with pred_path without tiling
    test_result = engine.predict(inputs=test_path, axes="YX")
    assert test_result is not None

    # Test prediction with pred_path with tiling
    test_result = engine.predict(
        inputs=test_path, axes="YX", tile_shape=patch_size, overlaps=overlaps
    )
    assert test_result is not None

    # Create engine from checkpoint
    del engine
    second_engine = Engine(model_path=result_model_path)

    # Test training with larger than memory dataset
    second_engine.cfg.data.in_memory = False
    _ = second_engine.train(train_path, val_path)

    # Test training w/o patching
    # TODO: test training w/o patching

    # Test training supervised mode
    engine = Engine(config=supervised_configuration)
    _ = engine.train(
        train_path=train_path,
        val_path=val_path,
        train_target_path=train_path,
        val_target_path=val_path,
    )
    model_name = f"{engine.cfg.experiment_name}_best.pth"
    result_model_path = engine.cfg.working_directory / model_name

    assert result_model_path.exists()
