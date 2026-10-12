"""Plotting utility functions."""

from pathlib import Path


def get_plot_file_path(save_path: Path | str, file_name: str) -> Path:
    """Get the file path for saving a plot.

    Parameters
    ----------
    save_path : Path | str
        The directory or file path where the plot should be saved.
    file_name : str
        The name of the file to save the plot as.

    Returns
    -------
    Path
        The full file path for saving the plot.
    """
    _save_file = Path(save_path)
    if _save_file.is_dir():
        _save_file = _save_file / file_name

    return _save_file.resolve()
