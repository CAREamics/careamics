from pathlib import Path

import numpy as np

from careamics.config import create_advanced_n2v_config
from careamics.lightning.logger import CoLogger


def test_logger(tmp_path: Path):
    config = create_advanced_n2v_config(
        experiment_name="careamics_testing",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=32,
        num_epochs=5,
        use_n2v2=False,
        use_tensorboard=True,
        use_wandb=False,
    )

    logger = CoLogger(
        experiment_name="careamics_testing",
        work_dir=tmp_path,
        config=config,
        use_tensorboard=True,
        use_wandb=False,
        log_version=0,
        finalize_after_fit=False,
    )

    assert logger.root_dir == tmp_path / "logs"

    logger.log_hyperparams(config.model_dump(), 0)

    log_dirs = logger.log_dir
    assert "csv" in log_dirs
    assert "tensorboard" in log_dirs
    assert Path(log_dirs["tensorboard"]).exists()

    logger.log_metrics({"acc": 0.99})

    logger.log_images(
        key="Random",
        images=np.random.rand(5, 1, 64, 64),
        step=0,
        captions=[f"img_{i}" for i in range(5)],
    )

    logger.finish("success")

    assert logger.finalize_after_fit
