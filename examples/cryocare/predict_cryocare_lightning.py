"""Predict a paired CryoCARE tomogram with a saved Lightning checkpoint."""

from pathlib import Path

import lightning.pytorch as L
import numpy as np
import tifffile

from careamics.config import CryoCAREDataConfig, MeanStdConfig
from careamics.config.algorithms.cryocare_algorithm_config import CryoCAREAlgorithm
from careamics.lightning import CareamicsDataModule
from careamics.lightning.modules import CryoCAREModule
from careamics.lightning.prediction import convert_prediction


def main() -> None:
    """Load a CryoCARE checkpoint and save a paired prediction as TIFF."""
    example_dir = Path(__file__).parent
    output_dir = Path("cryocare_example_outputs")
    checkpoint_path = output_dir / "checkpoints" / "epoch=199-step=11000.ckpt"
    even_path = example_dir / "data" / "Tomo110_even.mrc"
    odd_path = example_dir / "data" / "Tomo110_odd.mrc"
    prediction_path = output_dir / "Tomo110_denoised.tiff"

    algorithm_config = CryoCAREAlgorithm()
    model = CryoCAREModule.load_from_checkpoint(
        checkpoint_path,
        algorithm_config=algorithm_config,
        map_location="cpu",
    )

    data_config = CryoCAREDataConfig(
        mode="predicting",
        data_type="mrc",
        axes="ZYX",
        tilt_axis="Y",
        patching={
            "name": "tiled",
            "patch_size": (72, 72, 72),
            "overlaps": (32, 32, 32),
        },
        batch_size=1,
        normalization=MeanStdConfig(
            per_channel=False,
            target_mode="same_as_input",
        ),
    )
    data_module = CareamicsDataModule(
        data_config=data_config,
        pred_data=even_path,
        pred_data_target=odd_path,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    prediction_batches = trainer.predict(model, datamodule=data_module)
    predictions, _ = convert_prediction(prediction_batches, tiled=True)

    if len(predictions) != 1:
        raise ValueError(
            f"Expected one predicted volume, got {len(predictions)} volumes."
        )

    prediction = np.asarray(predictions[0])
    if prediction.ndim != 3:
        raise ValueError(
            "Expected a restored 3D ZYX prediction, "
            f"but received shape {prediction.shape}."
        )

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        prediction_path,
        prediction.astype(np.float32),
        metadata={"axes": "ZYX"},
    )
    print(f"Saved prediction to {prediction_path}.")


if __name__ == "__main__":
    main()
