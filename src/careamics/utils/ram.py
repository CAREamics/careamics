import psutil


def get_ram_size() -> int:
    """
    Get RAM size in bytes.

    Returns
    -------
    int
        RAM size in mbytes.
    """
    return psutil.virtual_memory().total / 1024**2
