from pathlib import Path

import pytest
from typer.testing import CliRunner

from careamics.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("algorithm", ["care", "n2n", "n2v"])
def test_conf(tmp_path: Path, algorithm: str):
    config_path = tmp_path / "config.yaml"
    result = runner.invoke(
        app,
        [
            "conf",
            "-d",
            str(tmp_path),
            algorithm,
            "--experiment-name",
            "LevitatingFrog",
            "--axes",
            "YX",
            "--patch-size",
            "64",
            "64",
            "-1",
            "--batch-size",
            "1",
            "--num-epochs",
            "1",
        ],
    )
    assert config_path.is_file()
    assert result.exit_code == 0
