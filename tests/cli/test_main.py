import pytest
from typer.testing import CliRunner

from careamics.cli.main import app

runner = CliRunner()

def test_train():
    # TODO
    result = runner.invoke(app, [
            # Command line args
        ]
    )
    # assert result.exit_code == 0 # replace with this when test is written
    assert result.exit_code == 2 # exit code from incorrect/missing parameters