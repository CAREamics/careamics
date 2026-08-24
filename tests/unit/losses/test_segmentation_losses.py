"""Tests for segmentation losses."""

from contextlib import nullcontext as does_not_raise

import pytest
import torch

from careamics.losses.segmentation_losses import (
    CrossEntropyLoss,
    DiceCELoss,
    DiceLoss,
    get_seg_loss,
)

# --- Test utilities

LOSSES = [DiceCELoss, DiceLoss, CrossEntropyLoss]

BIN_CLASS_LABELS = torch.tensor([[0, 1], [0, 1]])

BIN_ONE_HOT = torch.tensor(
    [
        [[1, 0], [1, 0]],
        [[0, 1], [0, 1]],
    ]
)

MUL_CLASS_LABELS = torch.tensor([[0, 1], [0, 2]])

MUL_ONE_HOT = torch.tensor(
    [
        [[1, 0], [1, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [0, 1]],
    ]
)


def to_3D(tensor: torch.Tensor) -> torch.Tensor:
    """Turn a 2D label map or one-hot map into a small 3D volume."""
    if tensor.ndim == 2:
        return torch.stack([tensor, torch.rot90(tensor, k=1)], dim=0)
    return torch.stack([tensor, torch.rot90(tensor, k=1, dims=(1, 2))], dim=1)


def to_batch(tensor: torch.Tensor, batch_size: int = 1) -> torch.tensor:
    """Add batch dimension by repeating the same sample."""
    return tensor.unsqueeze(0).repeat(batch_size, *([1] * tensor.ndim))


def make_targets(class_labels: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """Create batched targets of shape (B, 1, ...)."""
    return to_batch(class_labels.unsqueeze(0), batch_size=batch_size)


def low_loss_logits(one_hot, logit=10.0):
    """Logits corresponding to a near-perfect prediction."""
    return one_hot * logit + (1 - one_hot) * -logit


def high_loss_logits(one_hot, logit=10.0):
    """Logits corresponding to a confidently wrong prediction."""
    return one_hot * -logit + (1 - one_hot) * logit


# --- Unit tests


@pytest.mark.parametrize(
    "loss_func,class_labels,one_hot_labels,low_threshold",
    [
        (DiceLoss, BIN_CLASS_LABELS, BIN_ONE_HOT, 1e-3),
        (DiceLoss, MUL_CLASS_LABELS, MUL_ONE_HOT, 1e-3),
        (DiceCELoss, BIN_CLASS_LABELS, BIN_ONE_HOT, 1e-2),
        (DiceCELoss, MUL_CLASS_LABELS, MUL_ONE_HOT, 1e-2),
        (CrossEntropyLoss, BIN_CLASS_LABELS, BIN_ONE_HOT, 1e-3),
        (CrossEntropyLoss, MUL_CLASS_LABELS, MUL_ONE_HOT, 1e-3),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("is_3D", [False, True])
def test_loss_perfect_prediction_is_low(
    loss_func, class_labels, one_hot_labels, low_threshold, batch_size, is_3D
):
    """Perfect logits should produce a very small loss."""
    loss = loss_func()

    inputs = one_hot_labels if not is_3D else to_3D(one_hot_labels)
    targets = class_labels if not is_3D else to_3D(class_labels)

    logits = to_batch(low_loss_logits(inputs), batch_size=batch_size)
    targets = make_targets(targets, batch_size=batch_size)

    loss_value = loss(logits, targets)

    assert loss_value < low_threshold


@pytest.mark.parametrize(
    "loss_func,class_labels,one_hot_labels",
    [
        (DiceLoss, BIN_CLASS_LABELS, BIN_ONE_HOT),
        (DiceLoss, MUL_CLASS_LABELS, MUL_ONE_HOT),
        (DiceCELoss, BIN_CLASS_LABELS, BIN_ONE_HOT),
        (DiceCELoss, MUL_CLASS_LABELS, MUL_ONE_HOT),
        (CrossEntropyLoss, BIN_CLASS_LABELS, BIN_ONE_HOT),
        (CrossEntropyLoss, MUL_CLASS_LABELS, MUL_ONE_HOT),
    ],
)
def test_loss_wrong_prediction_is_higher(loss_func, class_labels, one_hot_labels):
    """Wrong logits should produce a larger loss than perfect logits."""
    loss = loss_func()
    low_logits = to_batch(low_loss_logits(one_hot_labels))
    high_logits = to_batch(high_loss_logits(one_hot_labels))
    targets = make_targets(class_labels)

    low_loss = loss(low_logits, targets)
    high_loss = loss(high_logits, targets)

    assert low_loss < high_loss
    assert high_loss - low_loss > 0.1


@pytest.mark.parametrize(
    "class_labels,one_hot_labels",
    [
        (BIN_CLASS_LABELS, BIN_ONE_HOT),
        (MUL_CLASS_LABELS, MUL_ONE_HOT),
        (BIN_CLASS_LABELS, BIN_ONE_HOT),
        (MUL_CLASS_LABELS, MUL_ONE_HOT),
        (BIN_CLASS_LABELS, BIN_ONE_HOT),
        (MUL_CLASS_LABELS, MUL_ONE_HOT),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("is_3D", [False, True])
def test_dice_ce_sum(class_labels, one_hot_labels, batch_size, is_3D):
    """Test that DiceCE is the sum of Dice and CE."""
    dice_ce = DiceCELoss()
    ce = CrossEntropyLoss()
    dice = DiceLoss()

    inputs = one_hot_labels if not is_3D else to_3D(one_hot_labels)
    targets = class_labels if not is_3D else to_3D(class_labels)

    inp = to_batch(low_loss_logits(inputs), batch_size=batch_size)
    tar = make_targets(targets, batch_size=batch_size)

    assert dice_ce.forward(inp, tar) == ce.forward(inp, tar) + dice.forward(inp, tar)


@pytest.mark.parametrize(
    "loss_name, exp_class, exp_error",
    [
        # no error
        ("dice", DiceLoss, does_not_raise()),
        ("ce", CrossEntropyLoss, does_not_raise()),
        ("dice_ce", DiceCELoss, does_not_raise()),
        # error
        ("not_a_loss", None, pytest.raises(ValueError, match="Unsupported")),
    ],
)
def test_get_loss(loss_name, exp_class, exp_error):
    """Test loss factory."""
    with exp_error:
        loss = get_seg_loss(loss_name)
        assert isinstance(loss, exp_class)
