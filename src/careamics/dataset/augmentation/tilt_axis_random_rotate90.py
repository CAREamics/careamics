"""Patch transform applying random 90 degree rotations perpendicular to a tilt axis."""

from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from careamics.dataset.augmentation.transform import Transform


class TiltAxisRandomRotate90(Transform):
    """Rotate cubic CZYX patches in the plane perpendicular to a tilt axis.

    Parameters
    ----------
    tilt_axis : Literal["Z", "Y", "X"]
        Tilt axis.
    p : float, default=1.0
        Probability of applying the transform.
    seed : int | None, default=None
        Seed for the random number generator.
    """

    _ROTATION_AXES: ClassVar[dict[str, tuple[int, int]]] = {
        "Z": (-2, -1),
        "Y": (-3, -1),
        "X": (-3, -2),
    }

    def __init__(
        self,
        tilt_axis: Literal["Z", "Y", "X"],
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """Initialize the transform.

        Parameters
        ----------
        tilt_axis : Literal["Z", "Y", "X"]
            Tilt axis.
        p : float, default=1.0
            Probability of applying the transform.
        seed : int | None, default=None
            Seed for the random number generator.
        """
        if tilt_axis not in self._ROTATION_AXES:
            raise ValueError("tilt_axis must be one of 'Z', 'Y', or 'X'.")
        if p < 0 or p > 1:
            raise ValueError("Probability must be in [0, 1].")

        self.tilt_axis = tilt_axis
        self.p = p
        self.rng = np.random.default_rng(seed=seed)

    def __call__(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Apply the same random rotation to all spatial arrays.

        Parameters
        ----------
        patch : np.ndarray
            Input CZYX patch with equal Z, Y, and X dimensions.
        target : numpy.ndarray | None, default=None
            Target CZYX patch.
        **additional_arrays : numpy.ndarray
            Additional CZYX arrays to transform identically.

        Returns
        -------
        numpy.ndarray
            Transformed input.
        numpy.ndarray | None
            Transformed target.
        dict[str, numpy.ndarray]]
            Additional transformed arrays.
        """
        self._validate_patch(patch)
        if target is not None:
            self._validate_patch(target)
        for array in additional_arrays.values():
            self._validate_patch(array)

        if self.rng.random() > self.p:
            return patch, target, additional_arrays

        n_rot = int(self.rng.integers(0, 4))
        axes = self._ROTATION_AXES[self.tilt_axis]
        patch_transformed = self._apply(patch, n_rot, axes)
        target_transformed = (
            self._apply(target, n_rot, axes) if target is not None else None
        )
        additional_transformed = {
            key: self._apply(array, n_rot, axes)
            for key, array in additional_arrays.items()
        }
        return patch_transformed, target_transformed, additional_transformed

    @staticmethod
    def _validate_patch(patch: NDArray) -> None:
        """Validate that an array is a cubic CZYX patch.

        Parameters
        ----------
        patch : np.ndarray
            Array to validate.
        """
        if patch.ndim != 4:
            raise ValueError("Tilt-axis rotation requires a 4D CZYX patch.")
        if not (patch.shape[-3] == patch.shape[-2] == patch.shape[-1]):
            raise ValueError("Tilt-axis rotation requires a cubic spatial patch.")

    @staticmethod
    def _apply(patch: NDArray, n_rot: int, axes: tuple[int, int]) -> NDArray:
        """Apply a rotation and return a contiguous array.

        Parameters
        ----------
        patch : np.ndarray
            Array to rotate.
        n_rot : int
            Number of 90-degree rotations.
        axes : tuple[int, int]
            Axes around which to rotate.

        Returns
        -------
        np.ndarray
            Rotated contiguous array.
        """
        return np.ascontiguousarray(np.rot90(patch, k=n_rot, axes=axes))
