from careamics.config.architectures import ArchitectureConfig


def test_model_dump():
    """Test that architecture keyword is removed from the model dump."""
    model_params = {"architecture": "LeCorbusier"}
    model = ArchitectureConfig(**model_params)

    # dump model
    model_dict = model.model_dump()
    assert model_dict == {}
