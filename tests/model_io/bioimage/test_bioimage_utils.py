import pytest

from careamics.model_io.bioimage.bioimage_utils import _format_bmz_name, format_bmz_path


@pytest.mark.parametrize(
    "name, expected",
    [
        ("A_name-with_Character", "a_name-with_character"),
        ("A_name with_Character", "a_name_with_character"),
        ("A_name (with_Character)", "a_name__with_character_"),
    ],
)
def test_format_bmz_name(name, expected):
    """Test formatting the bioimage.io model name."""
    assert _format_bmz_name(name) == expected


@pytest.mark.parametrize(
    "path, name",
    [
        ("path/to/folder", "My-Name (is_Bond)"),
        ("path/to/file.tar", ""),
        ("", "You Know_Who"),
    ],
)
def test_format_bmz_path(tmp_path, path, name):
    """Test formatting the bioimage.io model filename."""
    path = tmp_path / path
    new_path = format_bmz_path(path, name)

    assert new_path.parent.exists()
    assert new_path.suffix == ".zip"

    if path.suffix == "":
        assert new_path.name == _format_bmz_name(name) + ".zip"
