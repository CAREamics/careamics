#!/usr/bin/env python
"""Configure N2V denoising for a 3D OME-NGFF light-sheet image.

Data from Hess, M., Caton, M., Kothari, M., Swedlow, J., Liberali, P., & Quintas Glasner
de Medeiros, G. (2026). Reference OME-Zarr for 3D time-lapse light-sheet microscopy with
nuclei tracking (Version v0.1) [Dataset]. Zenodo.
https://doi.org/10.5281/zenodo.22078388
"""

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
    file_list = pooch.retrieve(
        "https://zenodo.org/records/22078388/files/001-small-lowT.zip?download=1",
        known_hash=("5069b9606d646f810e9f0b899d98916d8366bfb14bc74765774c1d4df4a5fe80"),
        processor=pooch.Unzip(),
        path=Path(__file__).parent / "data",
    )

    for f in file_list:
        if f.endswith("raw.ome.zarr/zarr.json"):
            return Path(f).parent

    return f


def main() -> None:
    """Configure CAREamics for 3D light-sheet denoising.

    Returns
    -------
    None
        The CAREamist application is initialized.
    """
    root = Path(".") / "n2v_training"
    root.mkdir(exist_ok=True, parents=True)

    path_to_zarr = download_data()

    config = create_advanced_n2v_config(
        experiment_name="n2v_lightsheet_denoising",
        data_type="zarr",
        axes="TCZYX",
        patch_size=(16, 128, 128),
        batch_size=2,
        num_epochs=10,
        num_steps=100,
        n_channels=2,
        independent_channels=True,
    )
    careamist = CAREamist(config=config, work_dir=root)
    careamist.train(train_data=path_to_zarr)

    results = root / "results"
    results.mkdir(exist_ok=True, parents=True)

    plot_loss(careamist.get_losses(), save_path=results / "loss.png")

    # prediction
    careamist.predict_to_disk(
        pred_data=path_to_zarr, tile_size=(16, 128, 128), tile_overlap=(4, 48, 48)
    )


if __name__ == "__main__":
    main()
