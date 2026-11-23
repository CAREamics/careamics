"""Fixtures shared by configuration unit tests."""

from unittest.mock import Mock

import pytest

from careamics.config.validators import training_validators


@pytest.fixture
def checkpointing_warning_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Return a mock for checkpointing guidance log messages."""
    warning = Mock()
    monkeypatch.setattr(training_validators.logger, "warning", warning)
    return warning
