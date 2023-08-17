import pytest

from careamics_restoration.config import Configuration
from careamics_restoration.engine import Engine


def test_engine_init_errors():
    with pytest.raises(ValueError):
        Engine(config=None, config_path=None, model_path=None)

    with pytest.raises(TypeError):
        Engine(config="config", config_path=None, model_path=None)


def test_engine_predict_errors(minimum_config: dict):
    config = Configuration(**minimum_config)
    engine = Engine(config=config)

    with pytest.raises(ValueError):
        engine.predict(external_input=None, pred_path=None)

    config.data.mean = None
    config.data.std = None
    with pytest.raises(ValueError):
        engine.predict(external_input=None, pred_path="None")
