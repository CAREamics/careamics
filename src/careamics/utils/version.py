"""Version utility."""

import logging

from careamics import __version__

logger = logging.getLogger(__name__)


def get_careamics_version() -> str:
    """Get clean CAREamics version.

    This method returns the latest `Major.Minor.Patch` version of CAREamics, removing
    any local version identifier.

    Returns
    -------
    str
        Clean CAREamics version.
    """
    parts = __version__.split(".")

    # for local installs that do not detect the latest versions via tags
    # (typically our CI will install `0.X.devX<hash>.<other hash>` versions)
    if "dev" in parts[2]:
        parts[2] = "*"
        clean_version = ".".join(parts[:3])

        logger.warning(
            f"Your CAREamics version seems to be a locally modified version "
            f"({__version__}). The recorded version for loading models will be "
            f"{clean_version}, which may not exist. If you want to ensure "
            f"exporting the model with an existing version, please install the "
            f"closest CAREamics version from PyPI or conda-forge."
        )

    # Remove any local version identifier
    return ".".join(parts[:3])
