from typing import Any, Dict, Tuple

import numpy as np
from albumentations import DualTransform


class XYRandomRotate90(DualTransform):
    """Applies random 90 degree rotations to the YX axis.

    This transform expects (Z)YXC dimensions.

    Parameters
    ----------
    p : int, optional
        Probability to apply the transform, by default 0.5
    is_3D : bool, optional
        Whether the patches are 3D, by default False
    """

    def __init__(self, p: int = 0.5, is_3D: bool = False):
        """Constructor.

        Parameters
        ----------
        p : int, optional
            Probability to apply the transform, by default 0.5
        is_3D : bool, optional
            Whether the patches are 3D, by default False
        """
        super().__init__(p=p)

        self.is_3D = is_3D

        # rotation axes
        if is_3D:
            self.axes = (1, 2)
        else:
            self.axes = (0, 1)

    def get_params(self, **kwargs: Any) -> Dict[str, int]:
        """Get the transform parameters.

        Returns
        -------
        Dict[str, int]
            Transform parameters.
        """
        return {"n_rotations": np.random.randint(1, 4)}

    def apply(self, patch: np.ndarray, n_rotations: int, **kwargs: Any) -> np.ndarray:
        """Apply the transform to the image.

        Parameters
        ----------
        patch : np.ndarray
            Image or image patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        flip_axis : int
            Axis along which to flip the patch.
        """
        if len(patch.shape) == 3 and self.is_3D:
            raise ValueError(
                "Incompatible patch shape and dimensionality. ZYXC patch shape "
                "expected, but got YXC shape."
            )

        return np.ascontiguousarray(np.rot90(patch, k=n_rotations, axes=self.axes))

    def apply_to_mask(
        self, mask: np.ndarray, n_rotations: int, **kwargs: Any
    ) -> np.ndarray:
        """Apply the transform to the mask.

        Parameters
        ----------
        mask : np.ndarray
            Mask or mask patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        if len(mask.shape) != 4 and self.is_3D:
            raise ValueError(
                "Incompatible mask shape and dimensionality. ZYXC patch shape "
                "expected, but got YXC shape."
            )

        return np.ascontiguousarray(np.rot90(mask, k=n_rotations, axes=self.axes))

    def get_transform_init_args_names(self) -> Tuple[str]:
        """
        Get the transform arguments.

        Returns
        -------
        Tuple[str]
            Transform arguments.
        """
        return ("p", "is_3D")
