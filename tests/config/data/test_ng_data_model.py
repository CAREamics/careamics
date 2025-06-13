import pytest

from careamics.config.data.ng_data_model import NGDataConfig
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
def test_ng_data_model_strategy(patching_strategy):

    # Test the DataModel class
    data_model = NGDataConfig(
        data_type="array",
        axes="YX",
        patching=patching_strategy,
    )

    assert data_model.patching.name == patching_strategy["name"]
