"""Version utility."""

from careamics import __version__


def get_clean_version() -> str:
    """Get clean CAREamics version.

    This method returns the latest `Major.Minor.Patch` version of CAREamics, removing
    any local version identifier.

    Returns
    -------
    str
        Clean CAREamics version.
    """
    parts = __version__.split(".")  # Remove any local version identifier
    return ".".join(parts[:3])
