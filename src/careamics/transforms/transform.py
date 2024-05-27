"""A general parent class for transforms."""

from typing import Any


class Transform:
    """A general parent class for transforms."""

    def __call__(self, *args: Any, **kwwargs: Any) -> Any:
        """Apply the transform."""
        pass
