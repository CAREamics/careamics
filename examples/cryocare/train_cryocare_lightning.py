"""Train and predict with CryoCARE using the CAREamics Lightning API.

This example uses two synthetic 3D reconstructions as the even/odd input-target
pair. Replace the arrays below with MRC-backed data and the corresponding loading
configuration when running on real tomograms.
"""

from pathlib import Path

import lightning.pytorch as L
import numpy as np
import tifffile
from lightning.pytorch.callbacks import ModelCheckpoint

from careamics import __version__
from careamics.config import CryoCAREDataConfig, MeanStdConfig
from careamics.config.algorithms.cryocare_algorithm_config import CryoCAREAlgorithm
from careamics.config.lightning.training_configuration import TrainingConfig
from careamics.lightning import CareamicsDataModule, ConfigSaverCallback
from careamics.lightning.modules import CryoCAREModule
from careamics.lightning.prediction import convert_prediction


def make_synthetic_pair(
    shape: tuple[int, int, int], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create two independently noisy reconstructions of the same volume.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial shape of the synthetic volume in ZYX order.
    seed : int
        Seed used to generate the clean volume and independent noise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Synthetic even and odd reconstructions.
    """
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=shape).astype(np.float32)
    even = clean + rng.normal(scale=0.2, size=shape).astype(np.float32)
    odd = clean + rng.normal(scale=0.2, size=shape).astype(np.float32)
    return even, odd


def main() -> None:
    """Create synthetic data, train CryoCARE, and predict paired data."""
    output_dir = Path("cryocare_example_outputs")
    output_dir.mkdir(exist_ok=True)

    train_even = Path("data") / "Tomo110_even.mrc"
    train_odd = Path("data") / "Tomo110_odd.mrc"

    algorithm_config = CryoCAREAlgorithm()
    data_config = CryoCAREDataConfig(
        mode="training",
        data_type="mrc",
        axes="ZYX",
        tilt_axis="Y",
        validation_fraction=0.1,
        patching={"name": "stratified", "patch_size": (72, 72, 72)},
        batch_size=16,
        n_val_patches=0,
        normalization=MeanStdConfig(
            per_channel=False,
            target_mode="same_as_input",
        ),
    )
    training_config = TrainingConfig(
        trainer_params={
            "max_epochs": 200,
        },
        checkpoint_params={"save_last": True},
        early_stopping_params=None,
    )

    model = CryoCAREModule(algorithm_config)
    data_module = CareamicsDataModule(
        data_config=data_config,
        train_data=train_even,
        train_data_target=train_odd,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch:02d}-{step}",
            **training_config.checkpoint_params,
        ),
        ConfigSaverCallback(
            __version__,
            "cryocare_lightning_example",
            training_config,
            data_config,
        ),
    ]

    trainer = L.Trainer(
        callbacks=callbacks,
        default_root_dir=output_dir,
        **training_config.trainer_params,
    )
    trainer.fit(model, datamodule=data_module)

    # CryoCARE prediction averages the model output for both reconstructions. The
    # prediction datamodule therefore receives the even and odd volumes as a pair.
    prediction_data_config = data_config.convert_mode(
        "predicting",
        new_patch_size=(72, 72, 72),
    )
    prediction_data_module = CareamicsDataModule(
        data_config=prediction_data_config,
        pred_data=train_even,
        pred_data_target=train_odd,
    )
    prediction_batches = trainer.predict(model, datamodule=prediction_data_module)
    predictions, _ = convert_prediction(prediction_batches, tiled=False)
    prediction = predictions[0]

    prediction_path = output_dir / "Tomo110_denoised.tiff"
    tifffile.imwrite(
        prediction_path, prediction.astype(np.float32), metadata={"axes": "ZYX"}
    )
    print(f"Predicted {len(prediction_batches)} paired batches.")
    print(f"Saved prediction to {prediction_path}.")


if __name__ == "__main__":
    main()
