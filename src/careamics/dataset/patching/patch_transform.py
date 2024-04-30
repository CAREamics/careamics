from typing import List, Union

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
            all_transforms[transform.name](**transform.model_dump())
            for transform in patch_transforms
        ]

        return Aug.Compose(
            transforms,
            # apply image aug to "target"
            additional_targets={"target": "image"}
            if (with_target and normalize_mask)  # TODO check this
            else {},
        )
