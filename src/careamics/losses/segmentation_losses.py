"""Segmentation losses."""

from collections.abc import Callable
from typing import Union

import torch
import torch.nn.functional as F
from torch.nn import Module

from careamics.config.losses import (
    CELossConfig,
    DiceCELossConfig,
    DiceLossConfig,
)
from careamics.utils import get_device

SegmentationLoss = Union[DiceLossConfig, DiceCELossConfig, CELossConfig]


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


def _weights_to_tensor(
    class_weights: list[float] | torch.Tensor | None,
) -> torch.Tensor | None:
    """Convert class weights to a floating-point tensor on the runtime device.

    Parameters
    ----------
    class_weights : list[float] or torch.Tensor or None
        Class weights from configuration.

    Returns
    -------
    torch.Tensor or None
        Class weights as Tensor.
    """
    if class_weights is None:
        return None
    if isinstance(class_weights, torch.Tensor):
        return class_weights.to(dtype=torch.float32, device=get_device())
    return torch.tensor(class_weights, dtype=torch.float32, device=get_device())


class DiceLoss(Module):
    """Dice loss for binary and multi-class segmentation.

    Applies softmax activation to the model logits and computes Dice coefficient per
    class, then averages across classes.

    Parameters
    ----------
    class_weights : torch.Tensor, optional
        A manual rescaling weight given to each class.
    """

    def __init__(self, class_weights: list[float] | None = None) -> None:
        """Constructor.

        Parameters
        ----------
        class_weights : list[float] | None
            A manual rescaling weight given to each class.
        """
        super().__init__()
        self.weights = _weights_to_tensor(class_weights)

    def forward(
        self, inputs: torch.tensor, targets: torch.tensor, smooth: float = 1
    ) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : torch.Tensor
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

        probabilities = probabilities.flatten(2)
        targets = targets.flatten(2)

        intersection = (probabilities * targets).sum(dim=2)
        union = probabilities.sum(dim=2) + targets.sum(dim=2)
        dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

        if self.weights is not None:
            if self.weights.shape[0] != num_classes:
                raise ValueError(
                    f"Class weights must have length {num_classes}, got "
                    f"{self.weights.shape[0]}."
                )

            dice_per_class = dice_per_class * self.weights.to(dice_per_class.device)

        return 1 - dice_per_class.mean()


class DiceCELoss(Module):
    """Combined Dice and Cross-Entropy loss for binary and multi-class segmentation.

    Parameters
    ----------
    class_weights : torch.Tensor, default=None
        A manual rescaling weight given to each class for both losses.
    ce_weight : float, default=1.0
        Weight for the cross-entropy component.
    dice_weight : float, default=1.0
        Weight for the Dice loss component.
    """

    def __init__(
        self,
        class_weights: list[float] | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        class_weights : list[float] | None
            A manual rescaling weight given to each class for both losses.
        ce_weight : float, default=1.0
            Weight for the cross-entropy component.
        dice_weight : float, default=1.0
            Weight for the Dice loss component.
        """
        super().__init__()
        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.weights = _weights_to_tensor(class_weights)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self, inputs: torch.tensor, targets: torch.tensor, smooth: float = 1
    ) -> torch.Tensor:
        """Compute combined Dice and Cross-Entropy loss.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : torch.Tensor
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
            weight=self.weights,
            reduction="mean",
        )

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class CrossEntropyLoss(Module):
    """Cross-entropy loss for segmentation targets with singleton label channels.

    Parameters
    ----------
    class_weights : torch.Tensor, default=None
        A manual rescaling weight given to each class for both losses.
    include_background : bool, default=True
        Whether to include the background class in the Dice loss calculation.
    """

    def __init__(self, class_weights: list[float] | None) -> None:
        """Constructor.

        Parameters
        ----------
        class_weights : list[float] | None
            A manual rescaling weight given to each class.
        """
        super().__init__()
        self.weights = _weights_to_tensor(class_weights)

    def forward(
        self,
        inputs: torch.tensor,
        targets: torch.tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss from segmentation logits and targets.

        Parameters
        ----------
        inputs : torch.Tensor
            Predicted logits of shape (B, C, [Z], Y, X) where C is the number of
            classes, including background (C=2 for binary).
        targets : torch.Tensor
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
            inputs, target_indices, weight=self.weights, reduction="mean"
        )


def get_seg_loss(loss_config: SegmentationLoss) -> Callable:
    """Get loss function by name.

    Parameters
    ----------
    loss_config : DiceLossConfig or DiceCELossConfig or CELossConfig
        Loss configuration.

    Returns
    -------
    Callable
        Corresponding loss function.
    """
    parameters = loss_config.model_dump(exclude={"name"})

    if loss_config.name == "dice":
        return DiceLoss(**parameters)
    elif loss_config.name == "ce":
        return CrossEntropyLoss(**parameters)
    elif loss_config.name == "dice_ce":
        return DiceCELoss(**parameters)
    else:
        raise ValueError(f"Unsupported loss function: {loss_config.name}")
