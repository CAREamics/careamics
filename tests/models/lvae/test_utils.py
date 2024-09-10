import pytest
import torch

from careamics.models.lvae.utils import crop_img_tensor, pad_img_tensor


@pytest.mark.parametrize(
    "x, size",
    [
        (torch.zeros((1, 1, 64, 64)), (32, 32)),
        (torch.zeros((1, 1, 64, 64, 64)), (32, 32, 32)),
    ],
)
def test_crop_img_shape(x, size):
    res = crop_img_tensor(x, size)
    assert res.shape[2:] == size


@pytest.mark.parametrize(
    "x, size",
    [
        (torch.zeros((1, 1, 64, 64)), (65, 32)),
        (torch.zeros((1, 1, 64, 64, 64)), (65, 32, 32)),
    ],
)
def test_crop_img_assert(x, size):
    with pytest.raises(ValueError):
        crop_img_tensor(x, size)


@pytest.mark.parametrize(
    "x, size, expected",
    [
        (
            torch.arange(1, 17).reshape((1, 1, 4, 4)),
            (2, 2),
            torch.tensor([[[[6, 7], [10, 11]]]]),
        ),
        (
            torch.arange(1, 28).reshape((1, 1, 3, 3, 3)),
            (2, 2, 2),
            torch.tensor([[[[[1, 2], [4, 5]], [[10, 11], [13, 14]]]]]),
        ),
    ],
)
def test_crop_img_result(x, size, expected):
    res = crop_img_tensor(x, size)
    assert torch.equal(res, expected)


@pytest.mark.parametrize(
    "x, size",
    [
        (torch.zeros((1, 1, 64, 64)), (96, 96)),
        (torch.zeros((1, 1, 64, 64, 64)), (96, 96, 96)),
    ],
)
def test_pad_img_shape(x, size):
    res = pad_img_tensor(x, size)
    assert res.shape[2:] == size


@pytest.mark.parametrize(
    "x, size",
    [
        (torch.zeros((1, 1, 64, 64)), (65, 32)),
        (torch.zeros((1, 1, 64, 64, 64)), (65, 32, 32)),
    ],
)
def test_pad_img_assert(x, size):
    with pytest.raises(ValueError):
        pad_img_tensor(x, size)


@pytest.mark.parametrize(
    "x, size, expected",
    [
        (
            torch.arange(1, 10).reshape((1, 1, 3, 3)),
            (5, 5),
            torch.tensor(
                [
                    [
                        [
                            [0, 0, 0, 0, 0],
                            [0, 1, 2, 3, 0],
                            [0, 4, 5, 6, 0],
                            [0, 7, 8, 9, 0],
                            [0, 0, 0, 0, 0],
                        ]
                    ]
                ]
            ),
        ),
        (
            torch.arange(1, 9).reshape((1, 1, 2, 2, 2)),
            (4, 4, 4),
            torch.tensor(
                [
                    [
                        [
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                            [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]],
                            [[0, 0, 0, 0], [0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0]],
                            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                        ]
                    ]
                ]
            ),
        ),
    ],
)
def test_pad_img_result(x, size, expected):
    res = pad_img_tensor(x, size)
    assert torch.equal(res, expected)
