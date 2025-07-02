import numpy as np
import pytest

from careamics import CAREamist
from careamics.config import Configuration
from careamics.utils import cwd
from careamics.utils.lightning_utils import read_csv_logger


@pytest.mark.mps_gh_fail
def test_read_logger(tmp_path, minimum_n2v_configuration):

    config = Configuration(**minimum_n2v_configuration)
    config.training_config.lightning_trainer_config = {"max_epochs": 10}

    array = np.arange(32 * 32).reshape((32, 32))

    with cwd(tmp_path):
        careamist = CAREamist(config)
        careamist.train(train_source=array)

        losses = read_csv_logger(config.experiment_name, tmp_path / "csv_logs")

    assert len(losses) == 4
    for key in losses:
        assert (
            len(losses[key])
            == config.training_config.lightning_trainer_config["max_epochs"]
        )
