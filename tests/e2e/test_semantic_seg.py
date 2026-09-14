import numpy as np
import pytest
from tifffile import imread

from careamics import CAREamist
from careamics.config import create_seg_config

# --- Test utilities


def toy_data(n_fgnd_classes: int = 1, with_channels: bool = False):
    """Creates a toy dataset.

    Returns
    -------
    np.ndarray
        Train data
    np.ndarray
        Train data target
    np.ndarray
        Val data
    np.ndarray
        Val data target
    """
    assert n_fgnd_classes in {1, 2}

    shape = (2, 3, 16, 16) if with_channels else (2, 16, 16)
    tar_shape = (2, 16, 16)

    train_data = 50 + np.zeros(shape).astype(np.int16)
    train_data_tar = np.zeros(tar_shape).astype(np.int16)

    train_data[..., :8, :8] = 200
    train_data[..., 8:, 8:] = 200
    train_data_tar[..., :8, :8] = 1
    train_data_tar[..., 8:, 8:] = 1

    if n_fgnd_classes == 2:
        train_data[..., 2:6, 2:6] = 160
        train_data[..., 10:14, 10:14] = 160
        train_data_tar[..., 2:6, 2:6] = 2
        train_data_tar[..., 10:14, 10:14] = 2

    return train_data, train_data_tar, train_data, train_data_tar


# --- e2e tests


# TODO fix seed?
@pytest.mark.parametrize("n_classes", [1, 2])
@pytest.mark.parametrize("with_channels", [False, True])
def test_segmentation(tmp_path, n_classes, with_channels):
    """Test semantic segmenttion with various classes, with/without input channels."""

    train_data, train_data_tar, val_data, val_data_tar = toy_data(
        n_classes, with_channels
    )

    cfg = create_seg_config(
        experiment_name="test_seg",
        data_type="array",
        axes="SCYX" if with_channels else "SYX",
        batch_size=4,
        patch_size=(4, 4),
        num_epochs=2,
        n_channels_in=3 if with_channels else None,
        n_classes=n_classes,
    )

    careamist = CAREamist(config=cfg, work_dir=tmp_path)
    careamist.train(
        train_data=train_data,
        train_data_target=train_data_tar,
        val_data=val_data,
        val_data_target=val_data_tar,
    )

    # predict to memory
    pred, _ = careamist.predict(pred_data=train_data)
    assert pred[0].shape == train_data_tar.shape
    assert set(np.unique(pred[0])) == set(range(n_classes + 1))

    # predict to disk
    careamist.predict_to_disk(pred_data=train_data, prediction_dir=tmp_path)
    assert (tmp_path / "array_0.tiff").exists()
    array_from_disk = imread(tmp_path / "array_0.tiff")
    np.testing.assert_array_almost_equal(pred[0], array_from_disk)
