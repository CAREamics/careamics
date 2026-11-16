import numpy as np
import tifffile

from careamics.cli.main import main
from careamics.config import create_n2v_config
from careamics.config.utils.configuration_io import save_configuration


def test_train(tmp_path):
    train_path = tmp_path / "train.tiff"
    val_path = tmp_path / "val.tiff"
    tifffile.imwrite(train_path, np.arange(32 * 32).reshape(32, 32))
    tifffile.imwrite(val_path, np.arange(32 * 32).reshape(32, 32))

    config = create_n2v_config(
        experiment_name="test-train",
        data_type="tiff",
        axes="YX",
        patch_size=(32, 32),
        batch_size=2,
        num_epochs=1,
        num_steps=1,
    )
    config_path = save_configuration(config, tmp_path / "careamics.yaml")
    output_path = tmp_path / "output"

    main(
        [
            "train",
            "--output_path",
            str(output_path),
            "--config",
            str(config_path),
            "--train_data",
            str(train_path),
            "--val_data",
            str(val_path),
        ]
    )

    records = list(output_path.glob("checkpoints/*/checkpoint_record.json"))
    assert len(records) == 1
