from pathlib import Path

from careamics.config.config_filter import (
    paths_to_str,
    remove_default_optionals,
)


def test_paths_to_str():
    """Test paths_to_str."""
    dictionary = {
        "path1": Path("path1"),
        "path2": "path2",
        "path3": Path("path3"),
        "path4": 3,
    }
    dictionary = paths_to_str(dictionary)
    assert isinstance(dictionary["path1"], str)
    assert isinstance(dictionary["path2"], str)
    assert isinstance(dictionary["path3"], str)
    assert isinstance(dictionary["path4"], int)


def test_remove_default_optionals():
    """Test remove default optionals."""
    dictionary = {
        "key1": "value1",
        "key2": 2,
        "key3": "value3",
        "key4": 5.5,
    }
    default = {
        "key1": "value6",
        "key2": 2,
        "key3": "value3",
    }

    remove_default_optionals(dictionary, default)
    assert dictionary["key1"] == "value1"
    assert "key2" not in dictionary.keys()
    assert "key3" not in dictionary.keys()
    assert dictionary["key4"] == 5.5
