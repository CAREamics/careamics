"""Auto-mark all tests under tests/compat/ with the 'compat' marker."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add `compat` marker to every test collected from this directory."""
    compat_marker = pytest.mark.compat
    for item in items:
        # item.fspath is the absolute path of the test file
        if "tests/compat" in str(item.fspath):
            item.add_marker(compat_marker)
