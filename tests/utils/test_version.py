from careamics.utils.version import get_careamics_version


def test_get_clean_version():
    """Test that the result has three members (major, minor, patch) and that the last
    one does not include `dev`."""
    version_members = get_careamics_version().split(".")

    assert len(version_members) == 3
    assert "dev" not in version_members[-1]
