import numpy as np

from careamics.config.transformations import (
    N2VManipulateModel,
    NDFlipModel,
    NormalizeModel,
    XYRandomRotate90Model,
)
from careamics.transforms import Compose, NDFlip, Normalize, XYRandomRotate90


def test_empty_compose(ordered_array):
    array = ordered_array((4, 5, 5))
    source = array[2:, ...]
    target = array[:2, ...]

    # instantiate Compose
    compose = Compose([])

    # apply the composed transform
    source_transformed, target_transformed = compose(source, target)

    # check the results
    assert (source_transformed == source).all()
    assert (target_transformed == target).all()


def test_compose_with_target(ordered_array):
    seed = 24
    array = ordered_array((4, 5, 5))
    source = array[2:, ...]
    target = array[:2, ...]

    # transform lists
    transform_list = [NDFlip(seed=seed), XYRandomRotate90(seed=seed)]
    transform_list_pydantic = [
        NDFlipModel(name="NDFlip", seed=seed),
        XYRandomRotate90Model(name="XYRandomRotate90", seed=seed),
    ]

    # instantiate Compose
    compose = Compose(transform_list_pydantic)

    # apply the composed transform
    source_transformed, target_transformed = compose(source, target)

    # apply them manually
    t1_source, t1_target = transform_list[0](source, target)
    t2_source, t2_target = transform_list[1](t1_source, t1_target)

    # check the results
    assert (source_transformed == t2_source).all()
    assert (target_transformed == t2_target).all()


def test_compose_n2v(ordered_array):
    seed = 24
    array = ordered_array((2, 5, 5))

    # transform lists
    mean, std = 0.5, 0.5

    transform_list_pydantic = [
        NormalizeModel(mean=mean, std=std),
        NDFlipModel(seed=seed),
        XYRandomRotate90Model(seed=seed),
        N2VManipulateModel(),
    ]

    # apply the transforms
    normalize = Normalize(mean=mean, std=std)
    ndflip = NDFlip(seed=seed)
    xyrotate = XYRandomRotate90(seed=seed)
    array_aug, _ = xyrotate(*ndflip(*normalize(array)))

    # instantiate Compose
    compose = Compose(transform_list_pydantic)

    # apply the composed transform
    results = compose(array)
    assert len(results) == 3  # output of n2v_manipulate
    assert (results[1] == array_aug).all()
    assert (
        results[0][np.where(results[2] == 1)] != array_aug[np.where(results[2] == 1)]
    ).all()
    assert (
        results[0][np.where(results[2] != 1)] == array_aug[np.where(results[2] != 1)]
    ).all()
