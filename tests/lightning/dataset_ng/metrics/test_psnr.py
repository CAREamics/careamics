import numpy as np
import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics import MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from careamics.lightning.dataset_ng.metrics import (
    SIPSNR,
    SampleSIPSNR,
)
from careamics.lightning.dataset_ng.metrics.psnr import _normalise_range
from careamics.utils.metrics import scale_invariant_psnr


def create_toy_data(shape):
    rng = np.random.default_rng(42)

    gt = 50 + np.zeros(shape, dtype=np.float32)
    for ch in range(shape[1]):
        intensity = rng.integers(40, 100)

        for _ in range(50):
            b = rng.integers(0, shape[0])
            x = rng.integers(0, shape[-2] - 20)
            y = rng.integers(0, shape[-1] - 20)

            if len(shape) == 5:
                z = rng.integers(0, shape[2] - 5)
                gt[b, ch, z : z + 5, x : x + 20, y : y + 20] += rng.integers(
                    0, intensity
                )
            else:
                gt[b, ch, x : x + 20, y : y + 20] += rng.integers(0, intensity)

    noisy = gt + rng.normal(0, 15, size=shape).astype(np.float32)
    noisy[noisy < 0] = 0

    return torch.from_numpy(gt), torch.from_numpy(noisy)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 64, 64),
        (2, 1, 64, 64),
        # channels
        (1, 3, 64, 64),
        (2, 3, 64, 64),
    ],
)
def test_fix_range_batch_independence(shape):

    gts, preds = create_toy_data(shape)

    # pass the whole batch
    gts_fixed, preds_fixed = _normalise_range(gts, preds)

    # individual images
    for i in range(shape[0]):

        # keep singleton dim
        gt_fixed, pred_fixed = _normalise_range(gts[[i]], preds[[i]])

        np.testing.assert_almost_equal(
            gt_fixed.numpy(), gts_fixed[[i]].numpy(), decimal=4
        )
        np.testing.assert_almost_equal(
            pred_fixed.numpy(), preds_fixed[[i]].numpy(), decimal=4
        )

    # error raised when not enough dimensions
    with pytest.raises(ValueError):
        _normalise_range(gts[0], preds[0])


@pytest.mark.parametrize(
    "shape, batch_size",
    [
        ((1, 1, 64, 64), 1),  # BC(Z)YX
        ((2, 1, 64, 64), 2),
        ((4, 1, 64, 64), 1),
        ((6, 1, 64, 64), 2),
        # channels
        ((1, 3, 64, 64), 1),
        ((2, 3, 64, 64), 2),
        ((4, 3, 64, 64), 1),
        ((6, 3, 64, 64), 2),
        # Z
        ((1, 1, 8, 64, 64), 1),
        ((2, 1, 8, 64, 64), 2),
        ((4, 1, 8, 64, 64), 1),
        ((6, 1, 8, 64, 64), 2),
        # Z + channels
        ((1, 3, 8, 64, 64), 1),
        ((2, 3, 8, 64, 64), 2),
        ((4, 3, 8, 64, 64), 1),
        ((6, 3, 8, 64, 64), 2),
    ],
)
def test_global_sipsnr(shape, batch_size):

    gts, preds = create_toy_data(shape)
    eps = torch.finfo(preds.dtype).eps

    # create batches
    batches = [
        (
            gts[i * batch_size : (i + 1) * batch_size],
            preds[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(gts.shape[0] // batch_size)
    ]

    # expected value from skimage
    # for this we need to use the range adjusted images and the global data range
    gts_fixed, preds_fixed = _normalise_range(gts, preds)

    dims = tuple(range(2, len(shape)))
    gts_max = torch.amax(gts, dim=(0,) + dims)
    gts_min = torch.amin(gts, dim=(0,) + dims)
    data_range = gts_max - gts_min + eps

    expected_psnr_rng = np.mean(
        [
            [
                peak_signal_noise_ratio(
                    gts_fixed[i][c].numpy(),
                    preds_fixed[i][c].numpy(),
                    data_range=data_range[c].item(),
                )
                for c in range(shape[1])
            ]
            for i in range(gts.shape[0])
        ],
        axis=0,
    )

    # compute metrics over batches
    metrics = SIPSNR(n_channels=shape[1])
    for gt_batch, pred_batch in batches:
        metrics.update(pred_batch, gt_batch)

    sipsnr = metrics.compute()

    assert (metrics.glob_max == gts_max).all()
    assert (metrics.glob_min == gts_min).all()
    np.testing.assert_almost_equal(sipsnr.numpy(), expected_psnr_rng, decimal=4)


@pytest.mark.parametrize(
    "shape, batch_size",
    [
        ((1, 1, 64, 64), 1),  # BC(Z)YX
        ((2, 1, 64, 64), 2),
        ((4, 1, 64, 64), 1),
        ((6, 1, 64, 64), 2),
        # channels
        ((1, 3, 64, 64), 1),
        ((2, 3, 64, 64), 2),
        ((4, 3, 64, 64), 1),
        ((6, 3, 64, 64), 2),
        # Z
        ((1, 1, 8, 64, 64), 1),
        ((2, 1, 8, 64, 64), 2),
        ((4, 1, 8, 64, 64), 1),
        ((6, 1, 8, 64, 64), 2),
        # Z + channels
        ((1, 3, 8, 64, 64), 1),
        ((2, 3, 8, 64, 64), 2),
        ((4, 3, 8, 64, 64), 1),
        ((6, 3, 8, 64, 64), 2),
    ],
)
def test_global_psnr(shape, batch_size):

    gts, preds = create_toy_data(shape)
    eps = torch.finfo(preds.dtype).eps

    # create batches
    batches = [
        (
            gts[i * batch_size : (i + 1) * batch_size],
            preds[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(gts.shape[0] // batch_size)
    ]

    # expected value from skimage
    dims = tuple(range(2, len(shape)))
    gts_max = torch.amax(gts, dim=(0,) + dims)
    gts_min = torch.amin(gts, dim=(0,) + dims)
    data_range = gts_max - gts_min + eps

    expected_psnr_rng = np.mean(
        [
            [
                peak_signal_noise_ratio(
                    gts[i][c].numpy(),
                    preds[i][c].numpy(),
                    data_range=data_range[c].item(),
                )
                for c in range(shape[1])
            ]
            for i in range(gts.shape[0])
        ],
        axis=0,
    )

    # compute metrics over batches
    metrics = SIPSNR(n_channels=shape[1], use_scale_invariance=False)
    for gt_batch, pred_batch in batches:
        metrics.update(pred_batch, gt_batch)

    glob_psnr = metrics.compute()
    assert (metrics.glob_max == gts_max).all()
    assert (metrics.glob_min == gts_min).all()
    np.testing.assert_almost_equal(glob_psnr.numpy(), expected_psnr_rng, decimal=4)


@pytest.mark.parametrize(
    "shape, batch_size",
    [
        ((1, 1, 64, 64), 1),  # BC(Z)YX
        ((2, 1, 64, 64), 2),
        ((4, 1, 64, 64), 1),
        ((6, 1, 64, 64), 2),
        # channels
        ((1, 3, 64, 64), 1),
        ((2, 3, 64, 64), 2),
        ((4, 3, 64, 64), 1),
        ((6, 3, 64, 64), 2),
        # Z
        ((1, 1, 8, 64, 64), 1),
        ((2, 1, 8, 64, 64), 2),
        ((4, 1, 8, 64, 64), 1),
        ((6, 1, 8, 64, 64), 2),
        # Z + channels
        ((1, 3, 8, 64, 64), 1),
        ((2, 3, 8, 64, 64), 2),
        ((4, 3, 8, 64, 64), 1),
        ((6, 3, 8, 64, 64), 2),
    ],
)
def test_sample_sipsnr(shape, batch_size):

    gts, preds = create_toy_data(shape)
    eps = torch.finfo(preds.dtype).eps

    # create batches
    batches = [
        (
            gts[i * batch_size : (i + 1) * batch_size],
            preds[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(gts.shape[0] // batch_size)
    ]

    # expected value from skimage
    # for this we need to use the range adjusted images and the local data ranges
    gts_fixed, preds_fixed = _normalise_range(gts, preds)

    dims = tuple(range(2, len(shape)))
    gts_max = torch.amax(gts, dim=dims)  # min/max along spatial dims
    gts_min = torch.amin(gts, dim=dims)
    data_range = gts_max - gts_min + eps

    expected_psnr_skimage = np.mean(
        [
            [
                peak_signal_noise_ratio(
                    gts_fixed[i][c].numpy(),
                    preds_fixed[i][c].numpy(),
                    data_range=data_range[i][c].item(),  # local data range
                )
                for c in range(shape[1])
            ]
            for i in range(gts.shape[0])
        ],
        axis=0,
    )
    expected_psnr_original = np.mean(
        [
            [
                scale_invariant_psnr(
                    gts_fixed[i][c].numpy(),
                    preds_fixed[i][c].numpy(),
                )
                for c in range(shape[1])
            ]
            for i in range(gts.shape[0])
        ],
        axis=0,
    )

    # compute metrics over batches
    metrics = SampleSIPSNR(n_channels=shape[1])
    for gt_batch, pred_batch in batches:
        metrics.update(pred_batch, gt_batch)

    sipsnr = metrics.compute()
    np.testing.assert_almost_equal(sipsnr.numpy(), expected_psnr_skimage, decimal=4)
    np.testing.assert_almost_equal(sipsnr.numpy(), expected_psnr_original, decimal=4)


@pytest.mark.parametrize(
    "shape, batch_size",
    [
        ((1, 1, 64, 64), 1),  # BC(Z)YX
        ((2, 1, 64, 64), 2),
        ((4, 1, 64, 64), 1),
        ((6, 1, 64, 64), 2),
        # channels
        ((1, 3, 64, 64), 1),
        ((2, 3, 64, 64), 2),
        ((4, 3, 64, 64), 1),
        ((6, 3, 64, 64), 2),
        # Z
        ((1, 1, 8, 64, 64), 1),
        ((2, 1, 8, 64, 64), 2),
        ((4, 1, 8, 64, 64), 1),
        ((6, 1, 8, 64, 64), 2),
        # Z + channels
        ((1, 3, 8, 64, 64), 1),
        ((2, 3, 8, 64, 64), 2),
        ((4, 3, 8, 64, 64), 1),
        ((6, 3, 8, 64, 64), 2),
    ],
)
def test_sample_psnr(shape, batch_size):

    gts, preds = create_toy_data(shape)
    eps = torch.finfo(preds.dtype).eps

    # create batches
    batches = [
        (
            gts[i * batch_size : (i + 1) * batch_size],
            preds[i * batch_size : (i + 1) * batch_size],
        )
        for i in range(gts.shape[0] // batch_size)
    ]

    # expected value from skimage
    dims = tuple(range(2, len(shape)))
    gts_max = torch.amax(gts, dim=dims)  # min/max along spatial dims
    gts_min = torch.amin(gts, dim=dims)
    data_range = gts_max - gts_min + eps

    expected_psnr_skimage = np.mean(
        [
            [
                peak_signal_noise_ratio(
                    gts[i][c].numpy(),
                    preds[i][c].numpy(),
                    data_range=data_range[i][c].item(),  # local data range
                )
                for c in range(shape[1])
            ]
            for i in range(gts.shape[0])
        ],
        axis=0,
    )

    # compute metrics over batches
    metrics = SampleSIPSNR(n_channels=shape[1], use_scale_invariance=False)
    for gt_batch, pred_batch in batches:
        metrics.update(pred_batch, gt_batch)

    sample_psnr = metrics.compute()
    np.testing.assert_almost_equal(
        sample_psnr.numpy(), expected_psnr_skimage, decimal=4
    )

    # if not 3D and no channels, compare with torchmetrics
    if len(shape) == 4 and shape[1] == 1:
        expected_psnr_torchmetrics_lst = []
        for i in range(gts.shape[0]):
            psnr_metric = PeakSignalNoiseRatio(data_range=data_range[i].numpy())
            psnr_metric.update(preds[i].unsqueeze(0), gts[i].unsqueeze(0))
            expected_psnr_torchmetrics_lst.append(psnr_metric.compute().numpy())
        expected_psnr_torchmetrics = np.mean(expected_psnr_torchmetrics_lst, axis=0)
        np.testing.assert_almost_equal(
            sample_psnr.numpy(), expected_psnr_torchmetrics, decimal=4
        )


def test_torchmetrics_collection():
    """Test that the PSNR metrics can be used in a torchmetrics MetricCollection."""
    metrics = MetricCollection(
        {
            "glob_sipsnr": SIPSNR(n_channels=3),
            "loc_sipsnr": SampleSIPSNR(n_channels=3),
            "glob_psnr": SIPSNR(n_channels=3, use_scale_invariance=False),
            "sample_psnr": SampleSIPSNR(n_channels=3, use_scale_invariance=False),
        }
    )

    shape = (2, 3, 8, 64, 64)
    gts, preds = create_toy_data(shape)

    metrics.update(preds, gts)
    results = metrics.compute()

    assert "glob_sipsnr" in results
    assert "loc_sipsnr" in results
    assert "glob_psnr" in results
    assert "sample_psnr" in results
