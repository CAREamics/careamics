"""A general parent class for transforms."""

from typing import Any


class Transform:
    """A general parent class for transforms."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the transform.

        Parameters
        ----------
        *args : Any
            Arguments.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        Any
            Transformed data.
        """
        pass
