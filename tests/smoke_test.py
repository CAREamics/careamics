from pathlib import Path
from typing import Tuple

import numpy as np
import yaml

from careamics_restoration.config import Configuration
from careamics_restoration.engine import Engine


def dump_config(configuration: Configuration) -> Path:
    temp_dir = configuration.working_directory
    config_path = temp_dir / "test_config.yml"
    config_dict = configuration.model_dump()
    with open(config_path, "w") as config_file:
        yaml.dump(config_dict, config_file)
    return config_path


def test_is_engine_runnable(
    base_configuration: Configuration,
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
    engine.train(train_path, val_path)

    model_name = f"{engine.cfg.experiment_name}_best.pth"
    result_model_path = engine.cfg.working_directory / model_name

    assert result_model_path.exists()

    # Test prediction with external input
    test_image = np.random.rand(*image_size)
    # Predict only accepts 4D input for now
    test_image = test_image[None, None, ...]
    test_result = engine.predict(external_input=test_image)

    assert test_result is not None

    # Test prediction with pred_path without tiling
    test_result = engine.predict(external_input=None, pred_path=test_path)

    assert test_result is not None

    # Create engine from checkpoint
    del engine
    second_engine = Engine(model_path=result_model_path)
    second_engine.train(train_path, val_path)

    # Test prediction with pred_path with tiling
    second_engine.cfg.prediction.tile_shape = patch_size
    second_engine.cfg.prediction.overlaps = overlaps
    second_engine.cfg.prediction.use_tiling = True
    test_result = second_engine.predict(external_input=None, pred_path=test_path)
    assert test_result is not None
