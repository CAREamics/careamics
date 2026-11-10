"""Train and predict with CryoCARE through the CAREamist API."""

from datetime import datetime
from pathlib import Path

from careamics import CAREamist
from careamics.config.factories import create_advanced_cryocare_config


def main() -> None:
    """Train CryoCARE on an odd/even MRC pair and export an MRC prediction."""
    example_dir = Path(__file__).parent
    output_dir = Path("cryocare_example_outputs")
    prediction_dir = Path("predictions") / datetime.now().strftime("%Y%m%d_%H%M%S")

    even_path = example_dir / "data" / "Tomo110_even.mrc"
    odd_path = example_dir / "data" / "Tomo110_odd.mrc"
    prediction_path = output_dir / prediction_dir / "Tomo110_even.mrc"

    # The main training parameters are exposed here for easy adjustment.
    patch_size = (72, 72, 72)
    batch_size = 16
    num_epochs = 200
    validation_fraction = 0.1
    n_val_patches = 50

    config = create_advanced_cryocare_config(
        experiment_name="cryocare_example",
        data_type="mrc",
        axes="ZYX",
        patch_size=patch_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        tilt_axis="Y",
        validation_fraction=validation_fraction,
        n_val_patches=n_val_patches,
        normalization_params={"target_mode": "same_as_input"},
        checkpoint_params={"save_last": True},
    )

    careamist = CAREamist(config=config, work_dir=output_dir)
    careamist.train(
        train_data=even_path,
        train_data_target=odd_path,
    )

    careamist.predict_to_disk(
        pred_data=even_path,
        pred_data_target=odd_path,
        prediction_dir=prediction_dir,
        tile_size=patch_size,
        tile_overlap=(32, 32, 32),
        checkpoint="last",
        write_type="mrc",
    )
    print(f"Saved prediction to {prediction_path}.")


if __name__ == "__main__":
    main()
