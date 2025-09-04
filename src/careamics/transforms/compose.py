"""A class chaining transforms together."""

from typing import Union, cast

from numpy.typing import NDArray

from careamics.config.transformations import NORM_AND_SPATIAL_UNION

from .normalize import Normalize
from .transform import Transform
from .xy_flip import XYFlip
from .xy_random_rotate90 import XYRandomRotate90

ALL_TRANSFORMS = {
    "Normalize": Normalize,
    "XYFlip": XYFlip,
    "XYRandomRotate90": XYRandomRotate90,
}


def get_all_transforms() -> dict[str, type]:
    """Return all the transforms accepted by CAREamics.

    Returns
    -------
    dict
        A dictionary with all the transforms accepted by CAREamics, where the keys are
        the transform names and the values are the transform classes.
    """
    return ALL_TRANSFORMS


class Compose:
    """A class chaining transforms together.

    Parameters
    ----------
    transform_list : list[TransformModel]
        A list of dictionaries where each dictionary contains the name of a
        transform and its parameters.

    Attributes
    ----------
    _callable_transforms : Callable
        A callable that applies the transforms to the input data.
    """

    def __init__(self, transform_list: list[NORM_AND_SPATIAL_UNION]) -> None:
        """Instantiate a Compose object.

        Parameters
        ----------
        transform_list : list[NORM_AND_SPATIAL_UNION]
            A list of dictionaries where each dictionary contains the name of a
            transform and its parameters.
        """
        # retrieve all available transforms
        # TODO: correctly type hint get_all_transforms function output
        all_transforms: dict[str, type[Transform]] = get_all_transforms()

        # instantiate all transforms
        self.transforms: list[Transform] = [
            all_transforms[t.name](**t.model_dump()) for t in transform_list
        ]

    def _chain_transforms(
        self, patch: NDArray, target: NDArray | None
    ) -> tuple[NDArray | None, ...]:
        """Chain transforms on the input data.

        Parameters
        ----------
        patch : np.ndarray
            Input data.
        target : Optional[np.ndarray]
            Target data, by default None.

        Returns
        -------
        tuple[np.ndarray, Optional[np.ndarray]]
            The output of the transformations.
        """
        params: Union[tuple[NDArray, NDArray | None],] = (patch, target)

        for t in self.transforms:
            *params, _ = t(*params)  # ignore additional_arrays dict

        # avoid None values that create problems for collating
        # TODO: removing None should be handled in dataset, not here
        return tuple(p for p in params if p is not None)

    def _chain_transforms_additional_arrays(
        self,
        patch: NDArray,
        target: NDArray | None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Chain transforms on the input data, with additional arrays.

        Parameters
        ----------
        patch : np.ndarray
            Input data.
        target : Optional[np.ndarray]
            Target data, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        tuple[np.ndarray, Optional[np.ndarray]]
            The output of the transformations.
        """
        params = {"patch": patch, "target": target, **additional_arrays}

        for t in self.transforms:
            patch, target, additional_arrays = t(**params)
            params = {"patch": patch, "target": target, **additional_arrays}

        return patch, target, additional_arrays

    def __call__(
        self, patch: NDArray, target: NDArray | None = None
    ) -> tuple[NDArray, ...]:
        """Apply the transforms to the input data.

        Parameters
        ----------
        patch : np.ndarray
            The input data.
        target : Optional[np.ndarray], optional
            Target data, by default None.

        Returns
        -------
        tuple[np.ndarray, ...]
            The output of the transformations.
        """
        # TODO: solve casting Compose.__call__ ouput
        return cast(tuple[NDArray, ...], self._chain_transforms(patch, target))

    def transform_with_additional_arrays(
        self,
        patch: NDArray,
        target: NDArray | None = None,
        **additional_arrays: NDArray,
    ) -> tuple[NDArray, NDArray | None, dict[str, NDArray]]:
        """Apply the transforms to the input data, including additional arrays.

        Parameters
        ----------
        patch : np.ndarray
            The input data.
        target : Optional[np.ndarray], optional
            Target data, by default None.
        **additional_arrays : NDArray
            Additional arrays that will be transformed identically to `patch` and
            `target`.

        Returns
        -------
        NDArray
            The transformed patch.
        NDArray | None
            The transformed target.
        dict of {str, NDArray}
            Transformed additional arrays. Keys correspond to the keyword argument
            names.
        """
        return self._chain_transforms_additional_arrays(
            patch, target, **additional_arrays
        )
