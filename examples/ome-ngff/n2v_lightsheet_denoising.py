#!/usr/bin/env python
"""Configure N2V denoising for a 3D OME-NGFF light-sheet image."""

from pathlib import Path

import pooch

from careamics import CAREamist
from careamics.config.factories import create_advanced_n2v_config
from careamics.plotting import plot_loss


def download_data() -> Path:
    """Download example data.

    Returns
    -------
    Path
        Path to data.
    """
    return Path(
        pooch.retrieve(
            "https://zenodo.org/records/22078388/files/001-small-lowT.zip?download=1"
        )
    )


def main() -> None:
    """Configure CAREamics for 3D light-sheet denoising.

    Returns
    -------
    None
        The CAREamist application is initialized.
    """
    root = Path(".") / "n2v_training"
    root.mkdir(exist_ok=True, parents=True)

    path = download_data()

    config = create_advanced_n2v_config(
        experiment_name="n2v_lightsheet_denoising",
        data_type="zarr",
        axes="ZYX",
        patch_size=(16, 128, 128),
        batch_size=8,
        num_epochs=10,
        num_steps=100,
    )
    careamist = CAREamist(config=config, work_dir=root)
    careamist.train(train_data=path)

    results = root / "results"
    results.mkdir(exist_ok=True, parents=True)

    plot_loss(careamist.get_losses(), save_path=results / "loss.png")

    # prediction
    careamist.predict_to_disk(
        pred_data=path, tile_size=(16, 128, 128), tile_overlap=(4, 48, 48)
    )


if __name__ == "__main__":
    main()
