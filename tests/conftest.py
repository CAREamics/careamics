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
            "pixel_manipulation": "n2v",
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
                "parameters": {"factor": 0.5, "patience": 5, "mode": "min"},
            },
            "amp": {
                "toggle": False,
                "init_scale": 1024,
            },
            "data":
            {
                "path": "path/to/data",
                "ext": "tif",
                "num_files": 2,
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "num_patches": None,
                "batch_size": 8,
                "num_workers": 0,
                "augmentation": None,
            }
        },
        "evaluation":
        {
            "data":
            {
                "path": "path/to/data",
                "ext": "tif",
                "num_files": 1,
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "num_patches": None,
                "batch_size": 1,
                "num_workers": 0,
                "batch_size": 1,
                "num_workers": 0,
                "augmentation": None,
            },
            "metric": "psnr",
        },
        "prediction":
        {
            "data": 
            {
                "path": "path/to/data",
                "ext": "tif",
                "num_files": 1,
                "extraction_strategy": "sequential",
                "patch_size": [128, 128],
                "num_patches": None,
                "batch_size": 1,
                "num_workers": 0,
                "batch_size": 1,
                "num_workers": 0,
                "augmentation": None,
            },
            "overlap": [25, 25],
        }
    }

    return test_configuration
