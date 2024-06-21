"""Utility function to get RAM size."""

import psutil


def get_ram_size() -> int:
    """
    Get RAM size in mbytes.

    Returns
    -------
    int
        RAM size in mbytes.
    """
    return psutil.virtual_memory().available / 1024**2
