import pytest

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)


def default_patching(mode: str) -> dict:
    """Return default patching strategy based on mode."""
    if mode == "training":
        return {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]}
    elif mode == "validating":
        return {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]}
    elif mode == "predicting":
        return {"name": SupportedPatchingStrategy.WHOLE}
    else:
        raise ValueError(f"Unknown mode: {mode}")


@pytest.mark.parametrize(
    "patching_strategy, mode",
    [
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "training",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "validating",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "predicting",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "predicting",
        ),
    ],
)
def test_ng_data_config_strategy(patching_strategy, mode):

    # Test the DataModel class
    data_config = NGDataConfig(
        mode=mode,
        data_type="array",
        axes="YX",
        patching=patching_strategy,
    )

    assert data_config.patching.name == patching_strategy["name"]


@pytest.mark.parametrize(
    "patching_strategy, mode",
    [
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "validating",
        ),
        (
            {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
            "predicting",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "training",
        ),
        (
            {"name": SupportedPatchingStrategy.FIXED_RANDOM, "patch_size": [16, 16]},
            "predicting",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "training",
        ),
        (
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
            "validating",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "training",
        ),
        (
            {
                "name": SupportedPatchingStrategy.WHOLE,
            },
            "validating",
        ),
    ],
)
def test_ng_data_wrong_config_strategy(patching_strategy, mode):
    # Test the DataModel class
    with pytest.raises(ValueError):
        _ = NGDataConfig(
            mode=mode,
            data_type="array",
            axes="YX",
            patching=patching_strategy,
        )


@pytest.mark.parametrize(
    "axes, patching_strategy",
    [
        ("ZYX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]}),
        ("YX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16, 16]}),
        (
            "ZYX",
            {
                "name": SupportedPatchingStrategy.RANDOM,
                "patch_size": [16, 16],
            },
        ),
        (
            "SYX",
            {
                "name": SupportedPatchingStrategy.RANDOM,
                "patch_size": [16, 16, 16],
            },
        ),
    ],
)
def test_ng_dataset_invalid_axes_patch(axes, patching_strategy):

    with pytest.raises(ValueError):
        NGDataConfig(
            mode="training",
            data_type="array",
            axes=axes,
            patching=patching_strategy,
        )


@pytest.mark.parametrize(
    "channels, error",
    [
        # valid
        (None, False),
        ([1], False),
        ([0, 4, 6], False),
        ((0, 1), False),
        # validator changes to valid value
        (3, False),
        ([], False),
        ((), False),
        # not a sequence
        ("invalid", True),
        ([0, -1], True),
        ([0, 3.3], True),
        ([0, 3, 0], True),
    ],
)
def test_channels(channels, error):
    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                mode="predicting",
                data_type="array",
                axes="CYX",
                patching=default_patching("predicting"),
                channels=channels,
            )
    else:
        _ = NGDataConfig(
            mode="predicting",
            data_type="array",
            axes="CYX",
            patching=default_patching("predicting"),
            channels=channels,
        )


def test_propagate_seed():
    global_seed = 42
    config = NGDataConfig(
        mode="training",
        data_type="array",
        axes="CYX",
        patching=default_patching("training"),
        patch_filter={
            "name": "shannon",
            "threshold": 0.5,
        },
        transforms=[{"name": "XYFlip"}],
        seed=global_seed,
    )

    assert config.seed == global_seed
    assert config.patching.seed == global_seed
    assert config.patch_filter.seed == global_seed
    for transform in config.transforms:
        assert transform.seed == global_seed


@pytest.mark.parametrize(
    "filter_config, mode, error",
    [
        ({"name": "mask"}, "training", False),
        ({"name": "mask"}, "validating", True),
        ({"name": "mask"}, "predicting", True),
    ],
)
def test_validate_coord_filters(filter_config, mode, error):

    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                mode=mode,
                data_type="array",
                axes="CYX",
                patching=default_patching(mode),
                coord_filter=filter_config,
            )
    else:
        _ = NGDataConfig(
            mode=mode,
            data_type="array",
            axes="CYX",
            patching=default_patching(mode),
            coord_filter=filter_config,
        )
