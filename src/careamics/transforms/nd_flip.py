from typing import Any, Dict, Tuple

import numpy as np
from albumentations import DualTransform


class NDFlip(DualTransform):
    """Flip ND arrays on a single axis.
    
    This transform ignores singleton axes and randomly flips one of the other
    axes, to the exception of the last axis (channels).
    """

    def __init__(
            self, 
            p: float = 0.5, 
            is_3D: bool = False, 
            flip_z: bool = True
        ):
        super().__init__(p=p)

        self.is_3D = is_3D
        self.flip_z = flip_z

        # "flippable" axes
        if is_3D:
            self.axis_indices = [0, 1, 2] if flip_z else [1, 2]
        else:
            self.axis_indices = [0, 1]

    def get_params(self, **kwargs: Any) -> Dict[str, int]:
        return {
            "flip_axis": np.random.choice(self.axis_indices)
        }    

    def apply(self, patch: np.ndarray, flip_axis: int, **kwargs: Any) -> np.ndarray:
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
        
        return np.ascontiguousarray(np.flip(patch, axis=flip_axis))

    def apply_to_mask(
            self, mask: np.ndarray, flip_axis: int, **kwargs: Any
        ) -> np.ndarray:
        """Apply the transform to the mask.

        Parameters
        ----------
        mask : np.ndarray
            Mask or mask patch, 2D or 3D, shape (y, x, c) or (z, y, x, c).
        """
        if len(mask.shape) == 3 and self.is_3D:
            raise ValueError(
                "Incompatible mask shape and dimensionality. ZYXC patch shape "
                "expected, but got YXC shape."
            )

        return np.ascontiguousarray(np.flip(mask, axis=flip_axis))
   
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("is_3D", "flip_z")
