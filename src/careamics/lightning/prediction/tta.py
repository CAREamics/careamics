"""Test-time augmentation derived from the training augmentations."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

from careamics.config.augmentations import XYFlipConfig, XYRandomRotate90Config


class TTAVariant(NamedTuple):
    """Deterministic XY transform: rotation by `rot_k` x 90 degrees, then flip.

    Attributes
    ----------
    rot_k : int
        Number of 90 degree rotations in the XY plane.
    flip_dim : int or None
        Dimension to flip after rotation, None for no flip.
    """

    rot_k: int
    flip_dim: int | None


def get_tta_variants(augmentations: Sequence[Any]) -> list[TTAVariant]:
    """Get the deterministic transforms that the training augmentations can produce.

    Only `XYFlipConfig` and `XYRandomRotate90Config` with a non-zero probability
    are used. Other augmentations are ignored.

    Parameters
    ----------
    augmentations : sequence of augmentation configurations
        Training augmentations.

    Returns
    -------
    list of TTAVariant
        TTA variants, the first one is the identity.
    """
    rot_ks = [0]
    flip_dims: list[int] = []
    for aug in augmentations:
        if isinstance(aug, XYFlipConfig) and aug.p > 0:
            flip_dims = [dim for dim, on in ((-1, aug.flip_x), (-2, aug.flip_y)) if on]
        elif isinstance(aug, XYRandomRotate90Config) and aug.p > 0:
            rot_ks = [0, 1, 2, 3]

    # with rotations, a single flip axis already spans all dihedral transforms
    if len(rot_ks) > 1:
        flip_dims = flip_dims[:1]

    return [TTAVariant(k, None) for k in rot_ks] + [
        TTAVariant(k, dim) for dim in flip_dims for k in rot_ks
    ]


def predict_with_tta(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    variants: Sequence[TTAVariant],
) -> torch.Tensor:
    """Average the model outputs over the TTA variants of the input.

    Parameters
    ----------
    model : callable
        Model.
    x : torch.Tensor
        Input batch with YX as the last two dimensions.
    variants : sequence of TTAVariant
        TTA variants.

    Returns
    -------
    torch.Tensor
        Mean of the inverse-transformed model outputs.
    """
    total = _transform(model(_transform(x, variants[0])), variants[0], inverse=True)
    for variant in variants[1:]:
        total = total + _transform(model(_transform(x, variant)), variant, inverse=True)
    return total / len(variants)


def _transform(
    tensor: torch.Tensor, variant: TTAVariant, inverse: bool = False
) -> torch.Tensor:
    """Apply a TTA variant, or its inverse, to the last two dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor with YX as the last two dimensions.
    variant : TTAVariant
        Transform to apply.
    inverse : bool, default=False
        Whether to apply the inverse transform.

    Returns
    -------
    torch.Tensor
        Transformed tensor.
    """
    if not inverse:
        tensor = torch.rot90(tensor, k=variant.rot_k, dims=(-2, -1))
    if variant.flip_dim is not None:
        tensor = torch.flip(tensor, dims=[variant.flip_dim])
    if inverse:
        tensor = torch.rot90(tensor, k=-variant.rot_k, dims=(-2, -1))
    return tensor
