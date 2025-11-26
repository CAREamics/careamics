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
    "in_memory, data_type, error",
    [
        # accepted combinations
        (True, "array", False),
        (True, "tiff", False),
        (True, "custom", False),
        (False, "tiff", False),
        (False, "custom", False),
        (False, "zarr", False),
        (False, "czi", False),
        # defaults are valid
        (None, "array", False),
        (None, "tiff", False),
        (None, "custom", False),
        (None, "zarr", False),
        (None, "czi", False),
        # validation errors
        (False, "array", True),
        (True, "zarr", True),
        (True, "czi", True),
    ],
)
def test_ng_data_config_in_memory(in_memory, data_type, error):
    if error:
        with pytest.raises(ValueError):
            NGDataConfig(
                data_type=data_type,
                axes="YX" if data_type != "czi" else "SCYX",
                in_memory=in_memory,
                patching={"name": SupportedPatchingStrategy.WHOLE},
            )
    else:
        config = NGDataConfig(
            data_type=data_type,
            axes="YX" if data_type != "czi" else "SCYX",
            in_memory=in_memory,
            patching={"name": SupportedPatchingStrategy.WHOLE},
        )

        # if in_memory is None, check the default value
        if in_memory is None:
            if data_type in ("array", "tiff", "custom"):
                assert config.in_memory is True
            else:
                assert config.in_memory is False
