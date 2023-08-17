import pytest

from careamics_restoration.engine import Engine


def test_engine_init_errors():
    with pytest.raises(ValueError):
        Engine(config=None, config_path=None, model_path=None)
    with pytest.raises(TypeError):
        Engine(config="config", config_path=None, model_path=None)
