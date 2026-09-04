"""Segmentation losses."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module


def _targets_to_class_indices(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert segmentation targets to class indices.

    This method removes the C dimension, casts the labels to long for the loss
    calculation, and performs validation.

    Parameters
    ----------
    targets : torch.Tensor
        Target representing class labels with a singleton C dimension.
    num_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Target as class indices tensor.
    """
    if targets.shape[1] == 1:
        class_indices = targets[:, 0].long()
    else:
        raise ValueError(
            f"Target channel dimension must be of size 1 (class labels), got size "
            f"{targets.shape[1]}."
        )

    if class_indices.min() < 0 or class_indices.max() >= num_classes:
        raise ValueError(
            f"Target class values must be in [0, {num_classes - 1}], got values in "
            f"[{class_indices.min().item()}, {class_indices.max().item()}]."
        )

    return class_indices


def _targets_to_one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert singleton-channel class labels to one-hot encoding.

    Parameters
    ----------
    targets : torch.Tensor
        Target representing class labels with a singleton C dimension.
    num_classes : int
        Number of classes.

    Returns
    -------
    torch.Tensor
        Target as one-hot encoded tensor.
    """
    class_indices = _targets_to_class_indices(targets, num_classes)
    one_hot = F.one_hot(class_indices, num_classes=num_classes).movedim(-1, 1)

    return one_hot.float()


class DiceLoss(Module):
    """Dice loss for binary and multi-class segmentation.

    Applies softmax activation to the model logits and computes Dice coefficient per
    class, then averages across classes.

    Parameters
    ----------
    weight : Tensor, optional
        A manual rescaling weight given to each class.
    include_background : bool, default=True
        Whether to include the background class (class 0) in the loss calculation.
    """

    def __init__(self, weight=None, include_background=True) -> None:
        """Constructor.

        Parameters
        ----------
        weight : Tensor, optional
            A manual rescaling weight given to each class.
        include_background : bool, default=True
            Whether to include the background class (class 0) in the loss calculation.
        """
        super().__init__()
        self.weight = weight
        self.include_background = include_background

    def forward(self, inputs, targets, smooth=1) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        inputs : Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : Tensor
            Ground truth of shape (B, 1, [Z], Y, X) with class indices.
        smooth : float, default=1
            Smoothing constant to avoid division by zero.

        Returns
        -------
        Tensor
            Dice loss value (1 - Dice coefficient).
        """
        num_classes = inputs.shape[1]

        probabilities = F.softmax(inputs, dim=1)
        targets = _targets_to_one_hot(targets, num_classes).to(inputs.device)

        if not self.include_background:
            probabilities = probabilities[:, 1:]
            targets = targets[:, 1:]

        probabilities = probabilities.flatten(2)
        targets = targets.flatten(2)

        intersection = (probabilities * targets).sum(dim=2)
        union = probabilities.sum(dim=2) + targets.sum(dim=2)
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

        if self.weight is not None:
            weight = self.weight
            if weight.shape[0] != num_classes:
                raise ValueError(
                    f"Class weights must have length {num_classes}, got "
                    f"{weight.shape[0]}."
                )

            if not self.include_background:
                weight = weight[1:]

            dice_per_class = dice_per_class * weight.to(dice_per_class.device)

        return 1 - dice_per_class.mean()


class DiceCELoss(Module):
    """Combined Dice and Cross-Entropy loss for binary and multi-class segmentation.

    Parameters
    ----------
    weight : Tensor, default=None
        A manual rescaling weight given to each class for both losses.
    include_background : bool, default=True
        Whether to include the background class in the Dice loss calculation.
    ce_weight : float, default=1.0
        Weight for the cross-entropy component.
    dice_weight : float, default=1.0
        Weight for the Dice loss component.
    """

    def __init__(
        self, weight=None, include_background=True, ce_weight=1.0, dice_weight=1.0
    ) -> None:
        """Constructor.

        Parameters
        ----------
        weight : Tensor, default=None
            A manual rescaling weight given to each class for both losses.
        include_background : bool, default=True
            Whether to include the background class in the Dice loss calculation.
        ce_weight : float, default=1.0
            Weight for the cross-entropy component.
        dice_weight : float, default=1.0
            Weight for the Dice loss component.
        """
        super().__init__()
        self.dice_loss = DiceLoss(weight=weight, include_background=include_background)
        self.weight = weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, smooth=1) -> torch.Tensor:
        """Compute combined Dice and Cross-Entropy loss.

        Parameters
        ----------
        inputs : Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : Tensor
            Ground truth of shape (B, 1, [Z], Y, X) with class indices.
        smooth : float, default=1
            Smoothing constant for Dice loss.

        Returns
        -------
        Tensor
            Combined loss value.
        """
        num_classes = inputs.shape[1]

        target_indices = _targets_to_class_indices(targets, num_classes).to(
            inputs.device
        )

        # compute Dice loss
        dice_loss = self.dice_loss(inputs, targets, smooth=smooth)

        # compute cross entropy
        ce_loss = F.cross_entropy(
            inputs,
            target_indices,
            weight=self.weight,
            reduction="mean",
        )

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class CrossEntropyLoss(Module):
    """Cross-entropy loss for segmentation targets with singleton label channels.

    Parameters
    ----------
    weight : Tensor, default=None
        A manual rescaling weight given to each class for both losses.
    include_background : bool, default=True
        Whether to include the background class in the Dice loss calculation.
    """

    def __init__(self, weight=None, include_background=True) -> None:
        """Constructor.

        Parameters
        ----------
        weight : Tensor, optional
            A manual rescaling weight given to each class.
        """
        super().__init__()
        self.weight = weight
        self.include_background = include_background

    def forward(self, inputs, targets) -> torch.Tensor:
        """Compute cross-entropy loss from segmentation logits and targets.

        Parameters
        ----------
        inputs : Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : Tensor
            Ground truth of shape (B, 1, [Z], Y, X) with class indices.

        Returns
        -------
        Tensor
            Loss value.
        """
        target_indices = _targets_to_class_indices(targets, inputs.shape[1]).to(
            inputs.device
        )
        return F.cross_entropy(
            inputs, 
            target_indices, 
            weight=self.weight, 
            reduction="mean",
            ignore_index=-100 if self.include_background else 0
        )


def get_seg_loss(loss: str) -> Callable:
    """Get loss function by name.

    Parameters
    ----------
    loss : str
        Name of the loss function. Supported: "dice", "ce", "dice_ce".

    Returns
    -------
    Callable
        Corresponding loss function.
    """
    if loss == "dice":
        return DiceLoss(include_background=False)
    elif loss == "ce":
        return CrossEntropyLoss(include_background=False)
    elif loss == "dice_ce":
        return DiceCELoss(include_background=False)
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
