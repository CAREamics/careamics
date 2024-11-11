import numpy as np

from careamics import CAREamist, Configuration
from careamics.utils import cwd
from careamics.utils.lightning_utils import read_csv_logger


def test_read_logger(tmp_path, minimum_configuration):

    config = Configuration(**minimum_configuration)
    config.training_config.num_epochs = 10

    array = np.arange(32 * 32).reshape((32, 32))

    with cwd(tmp_path):
        careamist = CAREamist(config)
        careamist.train(train_source=array)

        losses = read_csv_logger(config.experiment_name, tmp_path / "csv_logs")

    assert len(losses["epoch"]) == config.training_config.num_epochs
    assert len(losses["train"]) == config.training_config.num_epochs
    assert len(losses["val"]) == config.training_config.num_epochs
