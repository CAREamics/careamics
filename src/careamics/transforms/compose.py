"""A class chaining transforms together."""

from typing import Callable, List, Optional, Tuple

import numpy as np

from careamics.config.data_model import TRANSFORMS_UNION

from .n2v_manipulate import N2VManipulate
from .nd_flip import NDFlip
from .normalize import Normalize
from .transform import Transform
from .xy_random_rotate90 import XYRandomRotate90

ALL_TRANSFORMS = {
    "Normalize": Normalize,
    "N2VManipulate": N2VManipulate,
    "NDFlip": NDFlip,
    "XYRandomRotate90": XYRandomRotate90,
}


def get_all_transforms() -> dict:
    """Return all the transforms accepted by CAREamics.

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS


class Compose:
    """A class chaining transforms together."""

    def __init__(self, transform_list: List[TRANSFORMS_UNION]) -> None:
        """Instantiate a Compose object.

        Parameters
        ----------
        transform_list : List[TRANSFORMS_UNION]
            A list of dictionaries where each dictionary contains the name of a
            transform and its parameters.
        """
        # retrieve all available transforms
        all_transforms = get_all_transforms()

        # instantiate all transforms
        transforms = [all_transforms[t.name](**t.model_dump()) for t in transform_list]

        self._callable_transforms = self._chain_transforms(transforms)

    def _chain_transforms(self, transforms: List[Transform]) -> Callable:
        """Chain the transforms together.

        Parameters
        ----------
        transforms : List[Transform]
            A list of transforms to chain together.

        Returns
        -------
        Callable
            A callable that applies the transforms in order to the input data.
        """

        def _chain(
            patch: np.ndarray, target: Optional[np.ndarray]
        ) -> Tuple[np.ndarray, ...]:
            params = (patch, target)

            for t in transforms:
                params = t(*params)

            return params

        return _chain

    def __call__(
        self, patch: np.ndarray, target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, ...]:
        """Apply the transforms to the input data.

        Parameters
        ----------
        patch : np.ndarray
            The input data.
        target : Optional[np.ndarray], optional
            Target data, by default None

        Returns
        -------
        Tuple[np.ndarray, ...]
            The output of the transformations.
        """
        return self._callable_transforms(patch, target)
