import pytest


@pytest.fixture
def test_config(tmpdir):
    test_configuration = {
        "experiment_name": "testing",
        "workdir": str(tmpdir),
        "algorithm": {
            "name": "myalgo",
            "loss": ["n2v", "pn2v"],
            "model": "UNet",
            "num_masked_pixels": 128,
            "patch_size": [64, 64],
        },
        "training": {
            "num_epochs": 100,
            "learning_rate": 0.0001,
            "optimizer": {
                "name": "Adam",
                "parameters": {
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.0005,
                    "amsgrad": True,
                },
            },
            "lr_scheduler": {
                "name": "ReduceLROnPlateau",
                "parameters": {"factor": 0.5, "patience": 5},
            },
        },
    }

    return test_configuration
