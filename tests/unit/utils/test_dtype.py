import numpy as np

from careamics.utils.dtype import convert_floats_to_float32


def test_convert_floats_to_float32_recursively() -> None:
    metadata = {
        "float": 1.5,
        "nested": {
            "list": [np.float64(2.5), 3, "value"],
            "tuple": (np.float16(4.5), None),
        },
    }

    converted = convert_floats_to_float32(metadata)

    assert isinstance(converted["float"], np.float32)
    assert isinstance(converted["nested"]["list"][0], np.float32)
    assert isinstance(converted["nested"]["tuple"][0], np.float32)
    assert converted["nested"]["list"][1:] == [3, "value"]
    assert converted["nested"]["tuple"][1] is None


def test_convert_floats_to_float32_does_not_mutate_input() -> None:
    metadata = {"nested": {"values": [1.5]}}

    converted = convert_floats_to_float32(metadata)

    assert isinstance(metadata["nested"]["values"][0], float)
    assert converted is not metadata
    assert converted["nested"] is not metadata["nested"]
    assert converted["nested"]["values"] is not metadata["nested"]["values"]
