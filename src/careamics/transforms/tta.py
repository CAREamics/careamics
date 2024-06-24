"""Test-time augmentations."""

from torch import Tensor, flip, mean, rot90, stack


class ImageRestorationTTA:
    """
    Test-time augmentation for image restoration tasks.

    The augmentation is performed using all 90 deg rotations and their flipped version,
    as well as the original image flipped.

    Tensors should be of shape SC(Z)YX.

    This transformation is used in the LightningModule in order to perform test-time
    augmentation.
    """

    def forward(self, input_tensor: Tensor) -> list[Tensor]:
        """
        Apply test-time augmentation to the input tensor.

        Parameters
        ----------
        input_tensor : Tensor
            Input tensor, shape SC(Z)YX.

        Returns
        -------
        list of torch.Tensor
            List of augmented tensors.
        """
        # axes: only applies to YX axes
        axes = (-2, -1)

        augmented = [
            # original
            input_tensor,
            # rotations
            rot90(input_tensor, 1, dims=axes),
            rot90(input_tensor, 2, dims=axes),
            rot90(input_tensor, 3, dims=axes),
            # original flipped
            flip(input_tensor, dims=(axes[0],)),
            flip(input_tensor, dims=(axes[1],)),
        ]

        # rotated once, flipped
        augmented.extend(
            [
                flip(augmented[1], dims=(axes[0],)),
                flip(augmented[1], dims=(axes[1],)),
            ]
        )

        return augmented

    def backward(self, x: list[Tensor]) -> Tensor:
        """Undo the test-time augmentation.

        Parameters
        ----------
        x : Any
            List of augmented tensors of shape SC(Z)YX.

        Returns
        -------
        Any
            Original tensor.
        """
        axes = (-2, -1)

        reverse = [
            # original
            x[0],
            # rotated
            rot90(x[1], -1, dims=axes),
            rot90(x[2], -2, dims=axes),
            rot90(x[3], -3, dims=axes),
            # original flipped
            flip(x[4], dims=(axes[0],)),
            flip(x[5], dims=(axes[1],)),
            # rotated once, flipped
            rot90(flip(x[6], dims=(axes[0],)), -1, dims=axes),
            rot90(flip(x[7], dims=(axes[1],)), -1, dims=axes),
        ]

        return mean(stack(reverse), dim=0)
