from careamics.cli.create_config_n2v import create_config_n2v
from careamics.config.n2v_configuration import N2VConfiguration
from careamics.config.utils.configuration_io import load_configuration


def test_create_n2v_config(tmp_path):
    create_config_n2v(
        [
            "--output_path",
            str(tmp_path),
            "--experiment_name",
            "test-n2v",
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
    assert isinstance(config, N2VConfiguration)
