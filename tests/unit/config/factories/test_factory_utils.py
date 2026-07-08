import pytest

from careamics.config.factories.factory_utils import assemble_augmentations
from careamics.config.support import SupportedAugmentation

# --- List of parameters re-used in tests

AUGS = [
    [],
    ["x_flip"],
    ["y_flip"],
    ["rotate_90"],
    ["x_flip", "y_flip"],
    ["x_flip", "rotate_90"],
    ["y_flip", "rotate_90"],
    ["x_flip", "rotate_90", "y_flip"],
]

SEED = [None, 42]


# --- Unit tests


@pytest.mark.parametrize("augs", AUGS)
@pytest.mark.parametrize("seed", SEED)
def test_assemble_augmentations(augs, seed):
    """Test the assemble_augmentations function."""
    xy_flip = "x_flip" in augs or "y_flip" in augs
    rot_90 = "rotate_90" in augs

    augmentations = assemble_augmentations(augmentations=augs, seed=seed)

    is_xy_flip = any(
        aug.name == SupportedAugmentation.XY_FLIP.value for aug in augmentations
    )
    is_rot_90 = any(
        aug.name == SupportedAugmentation.XY_RANDOM_ROTATE90.value
        for aug in augmentations
    )

    for aug in augmentations:
        assert aug.seed == seed

    assert is_xy_flip == xy_flip
    assert is_rot_90 == rot_90
