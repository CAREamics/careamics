"""Pytest configuration for CLI tests."""

from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    """Mark every test collected from this directory as a CLI test."""
    cli_tests = Path(__file__).parent
    for item in items:
        if cli_tests in Path(item.fspath).parents:
            item.add_marker(pytest.mark.cli)
