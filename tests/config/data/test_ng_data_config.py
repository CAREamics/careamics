import pytest

from careamics.config.data.ng_data_config import NGDataConfig
from careamics.config.support.supported_patching_strategies import (
    SupportedPatchingStrategy,
)


@pytest.mark.parametrize(
    "patching_strategy",
    [
        {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]},
        # {
        #     "name": SupportedPatchingStrategy.SEQUENTIAL,
        #     "patch_size": [16, 16],
        #     "overlap": [4, 4],
        # },
        {
            "name": SupportedPatchingStrategy.TILED,
            "patch_size": [16, 16],
            "overlaps": [4, 4],
        },
        {
            "name": SupportedPatchingStrategy.WHOLE,
        },
    ],
)
def test_ng_data_config_strategy(patching_strategy):

    # Test the DataModel class
    data_config = NGDataConfig(
        data_type="array",
        axes="YX",
        patching=patching_strategy,
    )

    assert data_config.patching.name == patching_strategy["name"]


@pytest.mark.parametrize(
    "axes, patching_strategy",
    [
        ("ZYX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16]}),
        ("YX", {"name": SupportedPatchingStrategy.RANDOM, "patch_size": [16, 16, 16]}),
        (
            "ZYX",
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16],
                "overlaps": [4, 4],
            },
        ),
        (
            "SYX",
            {
                "name": SupportedPatchingStrategy.TILED,
                "patch_size": [16, 16, 16],
                "overlaps": [4, 4, 4],
            },
        ),
    ],
)
def test_ng_dataset_invalid_axes_patch(axes, patching_strategy):

    with pytest.raises(ValueError):
        NGDataConfig(
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
                data_type="array",
                axes="CYX",
                patching={"name": SupportedPatchingStrategy.WHOLE},
                channels=channels,
            )
    else:
        _ = NGDataConfig(
            data_type="array",
            axes="CYX",
            patching={"name": SupportedPatchingStrategy.WHOLE},
            channels=channels,
        )
