import pytest


def pytest_collection_modifyitems(items):
    """Mark all tests in this subtree as end-to-end tests."""
    for item in items:
        item.add_marker(pytest.mark.e2e)
