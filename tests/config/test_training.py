import pytest
from pydantic import conlist


from careamics.config.training import AMP, Training


@pytest.mark.parametrize("init_scale", [512, 1024, 65536])
def test_amp_init_scale(init_scale: int):
    """Test AMP init_scale parameter."""
    amp = AMP(use=True, init_scale=init_scale)
    assert amp.init_scale == init_scale


@pytest.mark.parametrize("init_scale", [511, 1088, 65537])
def test_amp_wrong_init_scale(init_scale: int):
    """Test wrong AMP init_scale parameter."""
    with pytest.raises(ValueError):
        AMP(use=True, init_scale=init_scale)


def test_amp_wrong_values_by_assignments():
    """Test that wrong values cause an error during assignment."""
    amp = AMP(use=True, init_scale=1024)

    # use
    amp.use = False
    with pytest.raises(ValueError):
        amp.use = None

    with pytest.raises(ValueError):
        amp.use = 3

    # init_scale
    amp.init_scale = 512
    with pytest.raises(ValueError):
        amp.init_scale = "1026"


def test_amp_to_dict():
    """ "Test export to dict."""
    # all values in there
    vals = {"use": True, "init_scale": 512}
    amp = AMP(**vals).model_dump()
    assert amp == vals

    assert "use" in amp.keys()
    assert "init_scale" in amp.keys()

    # optional value not in there if not specified (default)
    vals = {"use": True}
    amp = AMP(**vals).model_dump()
    assert amp == vals

    assert "use" in amp.keys()
    assert "init_scale" not in amp.keys()

    # optional value not in there if provided as default
    vals = {"use": True, "init_scale": 1024}
    amp = AMP(**vals).model_dump()

    assert "use" in amp.keys()
    assert "init_scale" not in amp.keys()


@pytest.mark.parametrize("num_epochs", [1, 2, 4, 9000])
def test_training_num_epochs(minimum_config: dict, num_epochs: int):
    """Test that Training accepts num_epochs greater than 0."""
    training = minimum_config["training"]
    training["num_epochs"] = num_epochs

    training = Training(**training)
    assert training.num_epochs == num_epochs


@pytest.mark.parametrize("num_epochs", [-1, 0])
def test_training_wrong_num_epochs(minimum_config: dict, num_epochs: int):
    """Test that wrong number of epochs cause an error."""
    training = minimum_config["training"]
    training["num_epochs"] = num_epochs

    with pytest.raises(ValueError):
        Training(**training)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 9000])
def test_training_batch_size(minimum_config: dict, batch_size: int):
    """Test batch size greater than 0."""
    training = minimum_config["training"]
    training["batch_size"] = batch_size

    training = Training(**training)
    assert training.batch_size == batch_size


@pytest.mark.parametrize("batch_size", [-1, 0])
def test_training_wrong_batch_size(minimum_config: dict, batch_size: int):
    """Test that wrong batch size cause an error."""
    training = minimum_config["training"]
    training["batch_size"] = batch_size

    with pytest.raises(ValueError):
        Training(**training)


@pytest.mark.parametrize("patch_size", [[2, 2], [2, 4, 2], [32, 96]])
def test_training_patch_size(minimum_config: dict, patch_size: conlist):
    """Test patch size greater than 0."""
    training = minimum_config["training"]
    training["patch_size"] = patch_size

    training = Training(**training)
    assert training.patch_size == patch_size


@pytest.mark.parametrize(
    "patch_size",
    [
        [
            2,
        ],
        [2, 4, 2, 2],
        [1, 1],
        [2, 0],
        [33, 32],
    ],
)
def test_training_wrong_patch_size(minimum_config: dict, patch_size: conlist):
    """Test that wrong patch size cause an error."""
    training = minimum_config["training"]
    training["patch_size"] = patch_size

    with pytest.raises(ValueError):
        Training(**training)


@pytest.mark.parametrize("num_workers", [0, 1, 4, 9000])
def test_training_num_workers(complete_config: dict, num_workers: int):
    """Test batch size greater than 0."""
    training = complete_config["training"]
    training["num_workers"] = num_workers

    training = Training(**training)
    assert training.num_workers == num_workers


@pytest.mark.parametrize("num_workers", [-1, -2])
def test_training_wrong_num_workers(complete_config: dict, num_workers: int):
    """Test that wrong batch size cause an error."""
    training = complete_config["training"]
    training["num_workers"] = num_workers

    with pytest.raises(ValueError):
        Training(**training)


def test_training_wrong_values_by_assignments(complete_config: dict):
    """Test that wrong values cause an error during assignment."""
    training = Training(**complete_config["training"])

    # num_epochs
    training.num_epochs = 2
    with pytest.raises(ValueError):
        training.num_epochs = -1

    # batch_size
    training.batch_size = 2
    with pytest.raises(ValueError):
        training.batch_size = -1

    # patch_size
    training.patch_size = [2, 2]
    with pytest.raises(ValueError):
        training.patch_size = [5, 4]

    # augmentation
    training.augmentation = True
    with pytest.raises(ValueError):
        training.augmentation = None

    # use_wandb
    training.use_wandb = True
    with pytest.raises(ValueError):
        training.use_wandb = None

    # amp
    training.amp = AMP(use=True, init_scale=1024)
    with pytest.raises(ValueError):
        training.amp = "I don't want to use AMP."

    # num_workers
    training.num_workers = 2
    with pytest.raises(ValueError):
        training.num_workers = -1


def test_training_to_dict_minimum(minimum_config: dict):
    """Test that the minimum config get export to dict correctly."""
    training_minimum = Training(**minimum_config["training"]).model_dump()
    assert training_minimum == minimum_config["training"]

    # Mandatory fields are present
    assert "num_epochs" in training_minimum.keys()
    assert "patch_size" in training_minimum.keys()
    assert "batch_size" in training_minimum.keys()

    assert "optimizer" in training_minimum.keys()
    assert "name" in training_minimum["optimizer"].keys()
    # optional subfield
    assert "parameters" not in training_minimum["optimizer"].keys()

    assert "lr_scheduler" in training_minimum.keys()
    assert "name" in training_minimum["lr_scheduler"].keys()
    # optional subfield
    assert "parameters" not in training_minimum["lr_scheduler"].keys()

    assert "augmentation" in training_minimum.keys()

    # Optionals fields are absent
    assert "wandb" not in training_minimum.keys()
    assert "num_workers" not in training_minimum.keys()
    assert "amp" not in training_minimum.keys()


def test_training_to_dict_optionals(complete_config: dict):
    """Test that the optionals fields are omitted when default."""
    train_conf = complete_config["training"]
    train_conf["amp"] = AMP(use=False, init_scale=1024)
    train_conf["num_workers"] = 0
    train_conf["use_wandb"] = False

    training_complete = Training(**train_conf).model_dump()

    # Mandatory fields are present
    assert "num_epochs" in training_complete.keys()
    assert "patch_size" in training_complete.keys()
    assert "batch_size" in training_complete.keys()

    assert "optimizer" in training_complete.keys()
    assert "name" in training_complete["optimizer"].keys()
    assert "parameters" in training_complete["optimizer"].keys()

    assert "lr_scheduler" in training_complete.keys()
    assert "name" in training_complete["lr_scheduler"].keys()
    assert "parameters" in training_complete["lr_scheduler"].keys()

    assert "augmentation" in training_complete.keys()

    # Optionals fields
    assert "use_wandb" not in training_complete.keys()
    assert "num_workers" not in training_complete.keys()
    assert "amp" not in training_complete.keys()
