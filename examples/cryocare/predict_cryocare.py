"""Predict a paired CryoCARE tomogram with a saved CAREamics checkpoint."""

from datetime import datetime
from pathlib import Path

from careamics import CAREamist


def main() -> None:
    """Load a CryoCARE checkpoint and export a paired prediction as MRC."""
    example_dir = Path(__file__).parent
    output_dir = Path("cryocare_example_outputs")
    prediction_dir = Path("predictions") / datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / "checkpoints" / "last.ckpt"
    even_path = example_dir / "data" / "Tomo110_even.mrc"
    odd_path = example_dir / "data" / "Tomo110_odd.mrc"
    prediction_path = output_dir / prediction_dir / "Tomo110_even.mrc"

    careamist = CAREamist(
        checkpoint_path=checkpoint_path,
        work_dir=output_dir,
    )
    careamist.predict_to_disk(
        pred_data=even_path,
        pred_data_target=odd_path,
        prediction_dir=prediction_dir,
        tile_size=(72, 72, 72),
        tile_overlap=(32, 32, 32),
        write_type="mrc",
    )
    print(f"Saved prediction to {prediction_path}.")


if __name__ == "__main__":
    main()
