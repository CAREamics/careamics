from typing import Callable, List, Union

import albumentations as Aug

from careamics.config.data_model import TRANSFORMS_UNION
from careamics.transforms import get_all_transforms


# TODO add some explanations on how the additional_targets is used
def get_patch_transform(
    patch_transforms: Union[List[TRANSFORMS_UNION], Aug.Compose],
    with_target: bool,
    normalize_mask: bool = True,
) -> Aug.Compose:
    """Return a pixel manipulation function."""
    # if we passed a Compose, we just return it
    if isinstance(patch_transforms, Aug.Compose):
        return patch_transforms

    # empty list of transforms is a NoOp
    elif len(patch_transforms) == 0:
        return Aug.Compose(
            [Aug.NoOp()],
            additional_targets={},  # TODO this part need be checked (wrt segmentation)
        )

    # else we have a list of transforms
    else:
        # retrieve all transforms
        all_transforms = get_all_transforms()

        # instantiate all transforms
        transforms = [
            all_transforms[transform.name](
                **transform.parameters.model_dump()
            )
            for transform in patch_transforms
        ]

        return Aug.Compose(
            transforms,
            # apply image aug to "target"
            additional_targets={"target": "image"}
            if (with_target and normalize_mask) # TODO check this
            else {},
        )


# TODO kept here as reference
def _get_patch_transform(
    patch_transforms: Union[List[TRANSFORMS_UNION], Aug.Compose],
    mean: float,
    std: float,
    target: bool,
    normalize_mask: bool = True,
) -> Aug.Compose:
    """Return a pixel manipulation function.

    Used in N2V family of algorithms.

    Parameters
    ----------
    patch_transform_type : str
        Type of patch transform.
    target : bool
        Whether the transform is applied to the target(if the target is present).
    mode : str
        Train or predict mode.

    Returns
    -------
    Union[None, Callable]
        Patch transform function.
    """
    if patch_transforms is None:
        return Aug.Compose(
            [Aug.NoOp()],
            additional_targets={"target": "image"}
            if (target and normalize_mask)  # TODO why? there is no normalization here?
            else {},
        )
    elif isinstance(patch_transforms, list):
        patch_transforms[[t["name"] for t in patch_transforms].index("Normalize")][
            "parameters"
        ] = {
            "mean": mean,
            "std": std,
            "max_pixel_value": 1,  # TODO why? mean/std normalization will not be lead to [-1,1] range
        }
        # TODO not very readable
        return Aug.Compose(
            [
                get_all_transforms()[transform["name"]](**transform["parameters"])
                if "parameters" in transform
                else get_all_transforms()[transform["name"]]()
                for transform in patch_transforms
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    elif isinstance(patch_transforms, Aug.Compose):
        return Aug.Compose(
            [
                t
                for t in patch_transforms.transforms[:-1]
                if not isinstance(t, Aug.Normalize)
            ]
            + [
                Aug.Normalize(mean=mean, std=std, max_pixel_value=1),
                patch_transforms.transforms[-1]
                if patch_transforms.transforms[-1].__class__.__name__ == "ManipulateN2V"
                else Aug.NoOp(),
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    else:
        raise ValueError(
            f"Incorrect patch transform type {patch_transforms}. "
            f"Please refer to the documentation."  # TODO add link to documentation
        )


# TODO add tta
def get_patch_transform_predict(
    patch_transforms: Union[List, Aug.Compose, None],
    mean: float,
    std: float,
    target: bool,
    normalize_mask: bool = True,
) -> Union[None, Callable]:
    """Return a pixel manipulation function.

    Used in N2V family of algorithms.

    Parameters
    ----------
    patch_transform_type : str
        Type of patch transform.
    target : bool
        Whether the transform is applied to the target(if the target is present).
    mode : str
        Train or predict mode.

    Returns
    -------
    Union[None, Callable]
        Patch transform function.
    """
    if patch_transforms is None:
        return Aug.Compose(
            [Aug.NoOp()],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    elif isinstance(patch_transforms, list):
        patch_transforms[[t["name"] for t in patch_transforms].index("Normalize")][
            "parameters"
        ] = {
            "mean": mean,
            "std": std,
            "max_pixel_value": 1,
        }
        # TODO not very readable
        return Aug.Compose(
            [
                get_all_transforms()[transform["name"]](**transform["parameters"])
                if "parameters" in transform
                else get_all_transforms()[transform["name"]]()
                for transform in patch_transforms
                if transform["name"] != "ManipulateN2V"
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    elif isinstance(patch_transforms, Aug.Compose):
        return Aug.Compose(
            [
                t
                for t in patch_transforms.transforms[:-1]
                if not isinstance(t, Aug.Normalize)
            ]
            + [
                Aug.Normalize(mean=mean, std=std, max_pixel_value=1),
            ],
            additional_targets={"target": "image"}
            if (target and normalize_mask)
            else {},
        )
    else:
        raise ValueError(
            f"Incorrect patch transform type {patch_transforms}. "
            f"Please refer to the documentation."  # TODO add link to documentation
        )
