"""Test prediction conversion and callback."""

import numpy as np
import pytest
import tifffile
import zarr
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from careamics.config import N2VAlgorithm, create_ng_data_configuration
from careamics.lightning.dataset_ng.callbacks.prediction_writer import (
    PredictionWriterCallback,
)
from careamics.lightning.dataset_ng.data_module import CareamicsDataModule
from careamics.lightning.dataset_ng.lightning_modules import N2VModule
from careamics.lightning.dataset_ng.prediction import convert_prediction

# TODO add zarr writing from tiff and arrays when it will be available
# TODO add untiled zarr writing when it will be implemented


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize(
    "shape, axes, channels, tiled",
    [
        ((32, 32), "YX", None, True),
        ((32, 32), "YX", None, False),
        ((3, 32, 32), "CYX", None, True),
        ((3, 32, 32), "CYX", None, False),
        ((3, 32, 32), "CYX", [1], True),
        ((3, 32, 32), "CYX", [1], False),
        ((3, 32, 32), "CYX", [0, 2], True),
        ((3, 32, 32), "CYX", [0, 2], False),
        ((16, 32, 32), "ZYX", None, True),
        ((16, 32, 32), "ZYX", None, False),
        ((5, 16, 32, 32), "SZYX", None, True),
        # ((5, 16, 32, 32), "SZYX", None, False),  # TODO: fails until #660 is fixed
        ((3, 16, 32, 32), "CZYX", None, True),
        ((3, 16, 32, 32), "CZYX", None, False),
        ((3, 16, 32, 32), "CZYX", [1], True),
        ((3, 16, 32, 32), "CZYX", [1], False),
        ((3, 16, 32, 32), "CZYX", [0, 2], True),
        ((3, 16, 32, 32), "CZYX", [0, 2], False),
        ((5, 3, 16, 32, 32), "SCZYX", None, True),
        # ((5, 3, 16, 32, 32), "SCZYX", None, False),  # TODO: fails until #660 is fixed
        ((5, 3, 16, 32, 32), "SCZYX", [1], True),
        # ((5, 3, 16, 32, 32), "SCZYX", [1], False),  # TODO: fails until #660 is fixed
        ((5, 3, 16, 32, 32), "SCZYX", [0, 2], True),
        # ((5, 3, 16, 32, 32), "SCZYX", [0, 2], False) # TODO: fails until #660 is fixed
    ],
)
def test_smoke_n2v_tiff(tmp_path, shape, axes, channels, tiled):
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, shape).astype(np.float32)
    val_array = rng.integers(0, 255, shape).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.tiff"
    train_file = train_dir / file_name
    tifffile.imwrite(train_file, train_array)

    if "C" in axes:
        len_channels = shape[axes.index("C")]
    else:
        len_channels = 1

    algorithm_config = N2VAlgorithm(
        model={
            "architecture": "UNet",
            "conv_dims": 2 if "Z" not in axes else 3,
            "in_channels": len(channels) if channels is not None else len_channels,
            "num_classes": len(channels) if channels is not None else len_channels,
        }
    )
    # create NGDataset configuration
    dataset_cfg = create_ng_data_configuration(
        data_type="array",
        axes=axes,
        patch_size=(16, 16) if "Z" not in axes else (8, 16, 16),
        batch_size=2,
        channels=channels,
    )

    # create lightning module
    model = N2VModule(algorithm_config=algorithm_config)

    # create data module
    data = CareamicsDataModule(
        data_config=dataset_cfg,
        train_data=train_array,
        val_data=val_array,
    )

    # create prediction writer callback params
    dirpath = tmp_path / "predictions"
    predict_writer = PredictionWriterCallback(dirpath=dirpath)

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            predict_writer,
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    predict_writer.set_writing_strategy(write_type="tiff", tiled=tiled)
    if tiled:
        pred_dataset_cfg = dataset_cfg.convert_mode(
            new_mode="predicting",
            new_data_type="tiff",
            new_batch_size=4,
            new_patch_size=(16, 16) if "Z" not in axes else (8, 16, 16),
            overlap_size=(4, 4) if "Z" not in axes else (4, 4, 4),
        )
    else:
        pred_dataset_cfg = dataset_cfg.convert_mode(
            new_mode="predicting",
            new_data_type="tiff",
            new_batch_size=4,
        )

    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=train_dir,
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=tiled)

    # assert predicted file exists
    assert (dirpath / file_name).is_file()
    save_data = tifffile.imread(dirpath / file_name)

    # note: predicted image is a list
    np.testing.assert_array_equal(save_data, predicted_images[0], verbose=True)


@pytest.mark.mps_gh_fail
@pytest.mark.parametrize(
    "shape, axes, channels",
    [
        ((32, 32), "YX", None),
        ((3, 32, 32), "CYX", None),
        ((3, 32, 32), "CYX", [1]),
        ((3, 32, 32), "CYX", [0, 2]),
        ((16, 32, 32), "ZYX", None),
        ((5, 16, 32, 32), "SZYX", None),
        ((3, 16, 32, 32), "CZYX", None),
        ((3, 16, 32, 32), "CZYX", [1]),
        ((3, 16, 32, 32), "CZYX", [0, 2]),
        ((5, 3, 16, 32, 32), "SCZYX", None),
        ((5, 3, 16, 32, 32), "SCZYX", [1]),
        ((5, 3, 16, 32, 32), "SCZYX", [0, 2]),
    ],
)
def test_smoke_n2v_zarr(tmp_path, shape, axes, channels):
    rng = np.random.default_rng(42)

    # training data
    train_array = rng.integers(0, 255, shape).astype(np.float32)
    val_array = rng.integers(0, 255, shape).astype(np.float32)

    # write train array to tiff
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    file_name = "image.zarr"
    g = zarr.open_group(train_dir / file_name, mode="w")
    arr = g.create_array("array", data=train_array, chunks="auto")

    if "C" in axes:
        len_channels = shape[axes.index("C")]
    else:
        len_channels = 1

    algorithm_config = N2VAlgorithm(
        model={
            "architecture": "UNet",
            "conv_dims": 2 if "Z" not in axes else 3,
            "in_channels": len(channels) if channels is not None else len_channels,
            "num_classes": len(channels) if channels is not None else len_channels,
        }
    )
    # create NGDataset configuration
    dataset_cfg = create_ng_data_configuration(
        data_type="array",
        axes=axes,
        patch_size=(16, 16) if "Z" not in axes else (8, 16, 16),
        batch_size=2,
        channels=channels,
    )

    # create lightning module
    model = N2VModule(algorithm_config=algorithm_config)

    # create data module
    data = CareamicsDataModule(
        data_config=dataset_cfg,
        train_data=train_array,
        val_data=val_array,
    )

    # create prediction writer callback params
    dirpath = tmp_path / "predictions"
    predict_writer = PredictionWriterCallback(dirpath=dirpath)

    # create trainer
    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "checkpoints",
                filename="test_lightning_api",
            ),
            predict_writer,
        ],
    )

    # train
    trainer.fit(model, datamodule=data)

    # predict
    predict_writer.set_writing_strategy(write_type="zarr", tiled=True)
    pred_dataset_cfg = dataset_cfg.convert_mode(
        new_mode="predicting",
        new_data_type="zarr",
        new_batch_size=4,
        new_patch_size=(16, 16) if "Z" not in axes else (8, 16, 16),
        overlap_size=(4, 4) if "Z" not in axes else (4, 4, 4),
        new_in_memory=False,
    )

    predict_data = CareamicsDataModule(
        data_config=pred_dataset_cfg,
        pred_data=[str(arr.store_path)],
    )

    # predict
    predicted = trainer.predict(model, datamodule=predict_data)
    predicted_images, _ = convert_prediction(predicted, tiled=True)

    # assert predicted file exists
    output_file = dirpath / f"{file_name[:-len('.zarr')]}_output.zarr"
    assert output_file.exists()
    save_data = zarr.open_array(output_file, path="array")

    predicted_img = predicted_images[0]

    # zarr writer removes singleton dims if not present in axes
    if "C" not in axes:
        predicted_img = predicted_img.squeeze(axis=1)
    if "S" not in axes:
        predicted_img = predicted_img.squeeze(axis=0)

    np.testing.assert_array_equal(save_data, predicted_img, verbose=True)
