"""
MicroSplit training and prediction script using CAREamics Lightning API.

This script trains a MicroSplit model for image splitting/denoising using
the CAREamics Lightning API with Gaussian Mixture Noise Models.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pooch
import tifffile
import torch
from careamics import CAREamist
from careamics.config import (
    GaussianMixtureNMConfig,
    create_microsplit_configuration,
    create_n2v_configuration,
)
from careamics.lightning import (
    create_microsplit_predict_datamodule,
    create_microsplit_train_datamodule,
)
from careamics.lightning.callbacks import DataStatsCallback
from careamics.lightning.lightning_module import VAEModule
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.eval_utils import get_device
from careamics.lvae_training.metrics import compute_stats
from careamics.models.lvae.noise_models import GaussianMixtureNoiseModel
from careamics.prediction_utils import convert_outputs_microsplit
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import get_train_val_data

DATA_URL = "https://download.fht.org/jug/msplit/ht_lif24/data_tiff/"
DATA_REGISTRY = {"ht_lif24_5ms_reduced.zip": None}
BASE_DIR = Path(__file__).resolve().parent  # folder containing this .py
DATA_DIR = BASE_DIR / "data"
NOISE_MODEL_DIR = BASE_DIR / "noise_models" / "5ms"
EXPERIMENT_NAME = "ht_lif24_5ms_reduced"
OUTPUT_DIR = Path("ht_lif24_reduced")

TRAIN_N2V = False
N2V_PATCH_SIZE = (64, 64)
N2V_BATCH_SIZE = 64
N2V_NUM_EPOCHS = 10

TRAIN_MICROSPLIT = True
PREDICT_MICROSPLIT = True

MICROSPLIT_PATCH_SIZE = (64, 64)
MICROSPLIT_BATCH_SIZE = 64
MICROSPLIT_NUM_EPOCHS = 10
MICROSPLIT_GRID_SIZE = 32
MICROSPLIT_Z_DIMS = [128] * 4
MICROSPLIT_ENCODER_N_FILTERS = 64
MICROSPLIT_DECODER_N_FILTERS = 64
MICROSPLIT_MULTISCALE_COUNT = 3
MICROSPLIT_OUTPUT_CHANNELS = 2
MICROSPLIT_MMSE_COUNT = 1
MICROSPLIT_NUM_WORKERS = 4

NM_N_COEFF = 3
NM_N_GAUSSIAN = 3
NM_N_EPOCHS = 1000

TRAINER_PRECISION = 16
TRAINER_GRADIENT_CLIP_VAL = 0.5
TRAINER_GRADIENT_CLIP_ALGORITHM = "value"

VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


def download_data() -> Path:
    """
    Download and extract the dataset using pooch.

    Returns
    -------
    Path
        Path to the extracted data directory.
    """
    data = pooch.create(
        path=DATA_DIR,
        base_url=DATA_URL,
        registry=DATA_REGISTRY,
    )
    for fname in data.registry:
        data.fetch(fname, processor=pooch.Unzip(), progressbar=True)

    data_path = data.abspath / (data.registry_files[0] + ".unzip/5ms/data/")
    return data_path


def prepare_noise_model_data(data_path: Path) -> np.ndarray:
    """
    Load and prepare data for noise model training.

    Parameters
    ----------
    data_path : Path
        Path to the data directory.

    Returns
    -------
    np.ndarray
        Training data for noise model.
    """
    nm_input = get_train_val_data(
        datadir=data_path,
        datasplit_type=DataSplitType.Train,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
    )
    return nm_input


def train_n2v_model(nm_input: np.ndarray) -> tuple[CAREamist, list[np.ndarray]]:
    """
    Train N2V model for denoising.

    Parameters
    ----------
    nm_input : np.ndarray
        Input data for training.

    Returns
    -------
    tuple[CAREamist, list[np.ndarray]]
        Trained CAREamist model and predictions.
    """
    config = create_n2v_configuration(
        experiment_name=EXPERIMENT_NAME,
        data_type="array",
        axes="SYXC",
        n_channels=2,
        patch_size=N2V_PATCH_SIZE,
        batch_size=N2V_BATCH_SIZE,
        num_epochs=N2V_NUM_EPOCHS,
    )

    careamist = CAREamist(source=config, work_dir=str(NOISE_MODEL_DIR))
    careamist.train(train_source=nm_input, val_minimum_split=5)

    prediction = careamist.predict(nm_input, tile_size=(256, 256))
    return careamist, prediction


def train_noise_models(
    nm_input: np.ndarray, prediction: list[np.ndarray]
) -> list[Path]:
    """
    Train Gaussian Mixture Noise Models for each channel.

    Parameters
    ----------
    nm_input : np.ndarray
        Original input data.
    prediction : list[np.ndarray]
        Denoised predictions from N2V.

    Returns
    -------
    list[Path]
        Paths to saved noise model files.
    """
    NOISE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nm_paths: list[Path] = []

    for channel_idx in range(nm_input.shape[-1]):
        channel_data = nm_input[..., channel_idx]
        channel_prediction = np.concatenate(prediction)[:, channel_idx]

        noise_model_config = GaussianMixtureNMConfig(
            model_type="GaussianMixtureNoiseModel",
            min_signal=float(channel_data.min()),
            max_signal=float(channel_data.max()),
            n_coeff=NM_N_COEFF,
            n_gaussian=NM_N_GAUSSIAN,
        )
        noise_model = GaussianMixtureNoiseModel(noise_model_config)
        noise_model.fit(
            signal=channel_data, observation=channel_prediction, n_epochs=NM_N_EPOCHS
        )

        nm_name = f"noise_model_Ch{channel_idx}"
        noise_model.save(path=str(NOISE_MODEL_DIR), name=nm_name)
        nm_paths.append(NOISE_MODEL_DIR / f"{nm_name}.npz")

    return nm_paths


def create_microsplit_config(nm_paths: list[Path]) -> Any:
    """
    Create MicroSplit configuration.

    Parameters
    ----------
    nm_paths : list[Path]
        Paths to noise model files.

    Returns
    -------
    Any
        MicroSplit configuration object.
    """
    config = create_microsplit_configuration(
        experiment_name=EXPERIMENT_NAME,
        data_type="tiff",
        axes="SYX",
        z_dims=MICROSPLIT_Z_DIMS,
        encoder_n_filters=MICROSPLIT_ENCODER_N_FILTERS,
        decoder_n_filters=MICROSPLIT_DECODER_N_FILTERS,
        patch_size=MICROSPLIT_PATCH_SIZE,
        grid_size=MICROSPLIT_GRID_SIZE,
        output_channels=MICROSPLIT_OUTPUT_CHANNELS,
        multiscale_count=MICROSPLIT_MULTISCALE_COUNT,
        batch_size=MICROSPLIT_BATCH_SIZE,
        num_epochs=MICROSPLIT_NUM_EPOCHS,
        mmse_count=MICROSPLIT_MMSE_COUNT,
        nm_paths=[str(p) for p in nm_paths],
        train_dataloader_params={"num_workers": MICROSPLIT_NUM_WORKERS},
        val_dataloader_params={"num_workers": MICROSPLIT_NUM_WORKERS},
        logger="wandb",
        trainer_params={
            "max_epochs": MICROSPLIT_NUM_EPOCHS,
            "precision": TRAINER_PRECISION,
            "gradient_clip_algorithm": TRAINER_GRADIENT_CLIP_ALGORITHM,
            "gradient_clip_val": TRAINER_GRADIENT_CLIP_VAL,
        },
    )
    return config


def load_pretrained_model(model: VAEModule, ckpt_path: str) -> None:
    """
    Load pretrained weights into a VAEModule.

    Parameters
    ----------
    model : VAEModule
        The model to load weights into.
    ckpt_path : str
        Path to the checkpoint file.
    """
    device = get_device()
    ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_dict["state_dict"], strict=False)


def create_trainer(config: Any) -> Trainer:
    """
    Create PyTorch Lightning Trainer with callbacks.

    Parameters
    ----------
    config : Any
        MicroSplit configuration object.

    Returns
    -------
    Trainer
        Configured PyTorch Lightning Trainer.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = OUTPUT_DIR / "checkpoints"

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename="ht_lif24_lightning_api",
            save_last=True,
        ),
        DataStatsCallback(),
    ]

    trainer = Trainer(
        max_epochs=config.training_config.lightning_trainer_config["max_epochs"],
        precision=config.training_config.lightning_trainer_config["precision"],
        gradient_clip_algorithm=config.training_config.lightning_trainer_config[
            "gradient_clip_algorithm"
        ],
        gradient_clip_val=config.training_config.lightning_trainer_config[
            "gradient_clip_val"
        ],
        default_root_dir=OUTPUT_DIR,
        callbacks=callbacks,
        logger=(
            WandbLogger(project="microsplit_lightning_api", name=EXPERIMENT_NAME)
            if config.training_config.logger == "wandb"
            else None
        ),
    )
    return trainer


def train_microsplit(
    config: Any, data_path: Path, pretrained_ckpt: Optional[str] = None
) -> tuple[VAEModule, Trainer, Any]:
    """
    Train MicroSplit model.

    Parameters
    ----------
    config : Any
        MicroSplit configuration object.
    data_path : Path
        Path to training data.
    pretrained_ckpt : Optional[str]
        Path to pretrained checkpoint to load, by default None.

    Returns
    -------
    tuple[VAEModule, Trainer, Any]
        Trained model, trainer, and data module.
    """
    model = VAEModule(config.algorithm_config)

    if pretrained_ckpt is not None:
        load_pretrained_model(model, pretrained_ckpt)

    train_data_module = create_microsplit_train_datamodule(
        train_data=data_path,
        patch_size=config.data_config.image_size,
        batch_size=config.data_config.batch_size,
        grid_size=config.data_config.grid_size,
        multiscale_count=config.data_config.multiscale_lowres_count,
        transforms=[],
        train_dataloader_params=config.data_config.train_dataloader_params,
        val_dataloader_params=config.data_config.val_dataloader_params,
    )

    trainer = create_trainer(config)
    trainer.fit(model, datamodule=train_data_module)

    return model, trainer, train_data_module


def predict_microsplit(
    model: VAEModule,
    trainer: Trainer,
    config: Any,
    train_data_module: Any,
    pred_data_path: str,
) -> np.ndarray:
    """
    Run prediction with trained MicroSplit model.

    Parameters
    ----------
    model : VAEModule
        Trained MicroSplit model.
    trainer : Trainer
        PyTorch Lightning Trainer.
    config : Any
        MicroSplit configuration object.
    train_data_module : Any
        Training data module (for data stats).
    pred_data_path : str
        Path to prediction data.

    Returns
    -------
    np.ndarray
        Stitched predictions.
    """
    data_stats, maxval = train_data_module.get_data_stats()

    predict_data_module = create_microsplit_predict_datamodule(
        pred_data=pred_data_path,
        tile_size=config.data_config.image_size,
        batch_size=config.data_config.batch_size,
        grid_size=config.data_config.grid_size,
        multiscale_count=config.data_config.multiscale_lowres_count,
        data_stats=data_stats,
        max_val=maxval,
        transforms=[],
        train_dataloader_params=config.data_config.train_dataloader_params,
        val_dataloader_params=config.data_config.val_dataloader_params,
    )

    predicted_tiles = trainer.predict(model, datamodule=predict_data_module)
    stitched_predictions, _ = convert_outputs_microsplit(
        predicted_tiles, predict_data_module.predict_dataset
    )

    return stitched_predictions


def evaluate_predictions(
    predictions: np.ndarray, highsnr_path: str, output_path: Optional[str] = None
) -> dict[str, Any]:
    """
    Evaluate predictions against high SNR ground truth.

    Parameters
    ----------
    predictions : np.ndarray
        Model predictions.
    highsnr_path : str
        Path to high SNR ground truth data.
    output_path : Optional[str]
        Path to save metrics JSON file.

    Returns
    -------
    dict[str, Any]
        Computed metrics.
    """
    highsnr_test_data = tifffile.imread(highsnr_path)
    metrics = compute_stats([highsnr_test_data[..., :2]], [predictions])

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main(
    skip_noise_model_training: bool = False,
    pretrained_ckpt: Optional[str] = None,
    pred_data_path: Optional[str] = None,
    highsnr_path: Optional[str] = None,
    output_metrics: Optional[str] = None,
) -> None:
    """
    Main function to run MicroSplit training and prediction pipeline.

    Parameters
    ----------
    skip_noise_model_training : bool
        If True, skip noise model training and use existing models.
    pretrained_ckpt : Optional[str]
        Path to pretrained checkpoint to load.
    pred_data_path : Optional[str]
        Path to prediction data.
    highsnr_path : Optional[str]
        Path to high SNR data for evaluation.
    output_metrics : Optional[str]
        Path to save metrics JSON file.
    """
    data_path = download_data()

    if skip_noise_model_training:
        nm_paths = [
            NOISE_MODEL_DIR / "noise_model_Ch0.npz",
            NOISE_MODEL_DIR / "noise_model_Ch1.npz",
        ]
    else:
        nm_input = prepare_noise_model_data(data_path)
        _, prediction = train_n2v_model(nm_input)
        nm_paths = train_noise_models(nm_input, prediction)

    config = create_microsplit_config(nm_paths)

    model, trainer, train_data_module = train_microsplit(
        config, data_path, pretrained_ckpt
    )

    if pred_data_path is not None:
        predictions = predict_microsplit(
            model, trainer, config, train_data_module, pred_data_path
        )

        if highsnr_path is not None:
            evaluate_predictions(predictions, highsnr_path, output_metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train and predict with MicroSplit using CAREamics Lightning API"
    )
    parser.add_argument(
        "--skip-noise-model",
        action="store_true",
        default=True,
        help="Skip noise model training and use existing models",
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Path to pretrained checkpoint to load",
    )
    parser.add_argument(
        "--pred-data",
        type=str,
        default=None,
        help="Path to prediction data directory",
    )
    parser.add_argument(
        "--highsnr-data",
        type=str,
        default=None,
        help="Path to high SNR data for evaluation",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None,
        help="Path to save metrics JSON file",
    )

    args = parser.parse_args()

    main(
        skip_noise_model_training=args.skip_noise_model,
        pretrained_ckpt=args.pretrained_ckpt,
        pred_data_path=args.pred_data,
        highsnr_path=args.highsnr_data,
        output_metrics=args.output_metrics,
    )
