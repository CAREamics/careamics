import pytest

from careamics.config import VAEAlgorithmConfig
from careamics.config.support import SupportedLoss


@pytest.mark.skip(
    reason="VAEAlgorithmConfig model is not currently serializable.\n"
    "The line `schema = VAEAlgorithmConfig.model_json_schema()` currently results "
    "in the following error:\n"
    "PydanticInvalidForJsonSchema: Cannot generate a JsonSchema for "
    "core_schema.IsInstanceSchema (<class 'torch.nn.modules.module.Module'>)"
)
def test_all_losses_are_supported():
    """Test that all losses defined in the Literal are supported."""
    # list of supported losses
    losses = list(SupportedLoss)

    # Algorithm json schema
    schema = VAEAlgorithmConfig.model_json_schema()

    # check that all losses are supported
    for loss in schema["properties"]["loss"]["enum"]:
        assert loss in losses
