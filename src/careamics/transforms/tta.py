"""Test-time augmentations."""

from typing import List

from torch import Tensor, flip, mean, rot90, stack


# TODO add tests
class ImageRestorationTTA:
    """
    Test-time augmentation for image restoration tasks.

    The augmentation is performed using all 90 deg rotations and their flipped version,
    as well as the original image flipped.

    Tensors should be of shape SC(Z)YX

    This transformation is used in the LightningModule in order to perform test-time
    agumentation.
    """

    def __init__(self) -> None:
        """Constructor."""
        pass

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Apply test-time augmentation to the input tensor.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape SC(Z)YX.

        Returns
        -------
        List[Tensor]
            List of augmented tensors.
        """
        augmented = [
            x,
            rot90(x, 1, dims=(-2, -1)),
            rot90(x, 2, dims=(-2, -1)),
            rot90(x, 3, dims=(-2, -1)),
        ]
        augmented_flip = augmented.copy()
        for x_ in augmented:
            augmented_flip.append(flip(x_, dims=(-3, -1)))
        return augmented_flip

    def backward(self, x: List[Tensor]) -> Tensor:
        """Undo the test-time augmentation.

        Parameters
        ----------
        x : Any
            List of augmented tensors.

        Returns
        -------
        Any
            Original tensor.
        """
        reverse = [
            x[0],
            rot90(x[1], -1, dims=(-2, -1)),
            rot90(x[2], -2, dims=(-2, -1)),
            rot90(x[3], -3, dims=(-2, -1)),
            flip(x[4], dims=(-3, -1)),
            rot90(flip(x[5], dims=(-3, -1)), -1, dims=(-2, -1)),
            rot90(flip(x[6], dims=(-3, -1)), -2, dims=(-2, -1)),
            rot90(flip(x[7], dims=(-3, -1)), -3, dims=(-2, -1)),
        ]
        return mean(stack(reverse), dim=0)
