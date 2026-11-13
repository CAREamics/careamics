from careamics.cli.main import main
from careamics.config.configuration import Configuration
from careamics.config.utils.configuration_io import load_configuration


def test_create_n2n_config(tmp_path):
    main(
        [
            "config-n2n",
            "--output_path",
            str(tmp_path),
            "--experiment_name",
            "test-n2n",
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
    assert isinstance(config, Configuration)
    assert config.algorithm_config.algorithm == "n2n"
