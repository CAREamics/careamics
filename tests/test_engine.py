import pytest

from careamics_restoration.config import Configuration
from careamics_restoration.engine import Engine
from careamics_restoration.models import create_model


def test_engine_init_errors():
    with pytest.raises(ValueError):
        Engine(config=None, config_path=None, model_path=None)

    with pytest.raises(TypeError):
        Engine(config="config", config_path=None, model_path=None)

    with pytest.raises(FileNotFoundError):
        Engine(config=None, config_path="some/path", model_path=None)

    with pytest.raises(FileNotFoundError):
        Engine(config=None, config_path=None, model_path="some/other/path")


def test_engine_predict_errors(minimum_config: dict):
    config = Configuration(**minimum_config)
    engine = Engine(config=config)

    with pytest.raises(ValueError):
        engine.predict(input=None)

    config.data.mean = None
    config.data.std = None
    with pytest.raises(ValueError):
        engine.predict(input="some/path")


@pytest.mark.parametrize(
    "epoch, losses", [(0, [1.0]), (1, [1.0, 0.5]), (2, [1.0, 0.5, 1.0])]
)
def test_engine_save_checkpoint(epoch, losses, minimum_config: dict):
    init_config = Configuration(**minimum_config)
    engine = Engine(config=init_config)

    # Mock engine attributes to test save_checkpoint
    engine.optimizer.param_groups[0]["lr"] = 1
    engine.lr_scheduler.patience = 1
    path = engine.save_checkpoint(epoch=epoch, losses=losses, save_method="state_dict")
    assert path.exists()

    if epoch == 0:
        assert path.stem.split("_")[-1] == "best"

    if losses[-1] == min(losses):
        assert path.stem.split("_")[-1] == "best"
    else:
        assert path.stem.split("_")[-1] == "latest"

    model, optimizer, scheduler, scaler, config = create_model(model_path=path)
    assert all(model.children()) == all(engine.model.children())
    assert optimizer.__class__ == engine.optimizer.__class__
    assert scheduler.__class__ == engine.lr_scheduler.__class__
    assert scaler.__class__ == engine.scaler.__class__
    assert optimizer.param_groups[0]["lr"] == engine.optimizer.param_groups[0]["lr"]
    assert optimizer.defaults["lr"] != engine.optimizer.param_groups[0]["lr"]
    assert scheduler.patience == engine.lr_scheduler.patience
    assert config == init_config
