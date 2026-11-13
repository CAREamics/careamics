from careamics.cli.create_config_hdn import create_config_hdn
from careamics.config.hdn_configuration import HDNConfiguration
from careamics.config.utils.configuration_io import load_configuration


def test_create_hdn_config(tmp_path):
    create_config_hdn(
        [
            "--output_path",
            str(tmp_path),
            "--experiment_name",
            "test-hdn",
            "--data_type",
            "array",
            "--axes",
            "YX",
            "--patch_size",
            "64",
            "64",
            "--batch_size",
            "8",
        ]
    )

    config = load_configuration(tmp_path / "careamics.yaml")
    assert isinstance(config, HDNConfiguration)
