import numpy as np
import pytest

from careamics.config.augmentations import (
    SwapInputTargetConfig,
    TiltAxisRandomRotate90Config,
)
from careamics.dataset.augmentation import SwapInputTarget, TiltAxisRandomRotate90
from careamics.dataset.augmentation.compose import Compose


@pytest.mark.parametrize(
    ("tilt_axis", "axes", "shape"),
    [
        ("Z", (-2, -1), (1, 5, 3, 3)),
        ("Y", (-3, -1), (1, 3, 5, 3)),
        ("X", (-3, -2), (1, 3, 3, 5)),
    ],
)
def test_tilt_axis_rotation(ordered_array, tilt_axis, axes, shape) -> None:
    patch = ordered_array(shape)
    target = patch + 100
    additional = patch + 200
    transform = TiltAxisRandomRotate90(tilt_axis, p=1, seed=42)
    rng = np.random.default_rng(42)
    rng.random()
    expected_k = int(rng.integers(0, 4))

    augmented, augmented_target, arrays = transform(patch, target, mask=additional)

    assert np.array_equal(augmented, np.rot90(patch, expected_k, axes=axes))
    assert np.array_equal(augmented_target, np.rot90(target, expected_k, axes=axes))
    assert np.array_equal(arrays["mask"], np.rot90(additional, expected_k, axes=axes))


def test_tilt_axis_rotation_requires_czyx_patch(ordered_array) -> None:
    transform = TiltAxisRandomRotate90("Z")

    with pytest.raises(ValueError, match="4D CZYX"):
        transform(ordered_array((1, 3, 3)))
    with pytest.raises(ValueError, match="equal dimensions"):
        transform(ordered_array((1, 2, 3, 4)))


def test_swap_input_target(ordered_array) -> None:
    patch = ordered_array((1, 2, 2, 2))
    target = patch + 10
    additional = patch + 20

    augmented, augmented_target, arrays = SwapInputTarget(p=1, seed=42)(
        patch, target, mask=additional
    )

    assert np.array_equal(augmented, target)
    assert np.array_equal(augmented_target, patch)
    assert np.array_equal(arrays["mask"], additional)


def test_cryo_care_augmentations_are_composable(ordered_array) -> None:
    patch = ordered_array((1, 2, 2, 2))
    target = patch + 10
    compose = Compose(
        [
            TiltAxisRandomRotate90Config(tilt_axis="Z", p=0),
            SwapInputTargetConfig(p=1),
        ]
    )

    augmented, augmented_target = compose(patch, target)

    assert np.array_equal(augmented, target)
    assert np.array_equal(augmented_target, patch)
