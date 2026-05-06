"""
Logging submodule.

The methods are responsible for the in-console logger.
"""

import logging
from pathlib import Path
from typing import Union

LOGGERS: dict = {}


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_path: Union[str, Path] | None = None,
) -> logging.Logger:
    """
    Create a python logger instance with configured handlers.

    Parameters
    ----------
    name : str
        Name of the logger.
    log_level : int, optional
        Log level (info, error etc.), by default logging.INFO.
    log_path : Optional[Union[str, Path]], optional
        Path in which to save the log, by default None.

    Returns
    -------
    logging.Logger
        Logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    if name in LOGGERS:
        return logger

    for logger_name in LOGGERS:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    if log_path:
        handlers = [
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ]
    else:
        handlers = [logging.StreamHandler()]

    formatter = logging.Formatter("%(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)  # type: ignore
        handler.setLevel(log_level)  # type: ignore
        logger.addHandler(handler)  # type: ignore

    logger.setLevel(log_level)
    LOGGERS[name] = True

    logger.propagate = False

    return logger
