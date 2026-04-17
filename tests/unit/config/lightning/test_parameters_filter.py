from careamics.config.lightning.parameters_filter import get_unknown_parameters


def a_test_func(param_1: str, param_2: int, param_3: float = 0.1):
    """Test function for _get_unknown_parameters."""
    pass


# ------------------------ Unit tests --------------------------


def test_get_unknown_parameters():
    """Test that _get_unknown_parameters returns the correct unknown parameters."""
    user_params = {
        "param_1": "test",
        "param_3": 0.2,
        "param_4": "unknown",
    }
    unknown_params = get_unknown_parameters(a_test_func, user_params)
    assert unknown_params == {"param_4": "unknown"}
