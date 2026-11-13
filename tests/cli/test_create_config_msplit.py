from careamics.cli.create_config_msplit import create_config_msplit
from careamics.config.microsplit_configuration import MicroSplitConfiguration
from careamics.config.utils.configuration_io import load_configuration


def test_create_msplit_config(tmp_path):
    create_config_msplit(
        [
            "--output_path",
            str(tmp_path),
            "--experiment_name",
            "test-msplit",
            "--data_type",
            "array",
            "--axes",
            "YX",
            "--patch_size",
            "64",
            "64",
            "--batch_size",
            "8",
            "--output_channels",
            "3",
        ]
    )

    config = load_configuration(tmp_path / "careamics.yaml")
    assert isinstance(config, MicroSplitConfiguration)
