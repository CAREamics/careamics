"""Utility function to get RAM size."""

import psutil


# TODO only used in compat, but we should use it for loading files in memory
def get_ram_size() -> int:
    """
    Get RAM size in mbytes.

    Returns
    -------
    int
        RAM size in mbytes.
    """
    return psutil.virtual_memory().available / 1024**2
