"""
Logging submodule.

The methods are responsible for the in-console logger.
"""

import logging
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union

LOGGERS: dict = {}


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_path: Optional[Union[str, Path]] = None,
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


class ProgressBar:
    """
    Keras style progress bar.

    Adapted from https://github.com/yueyericardo/pkbar.

    Parameters
    ----------
    max_value : Optional[int], optional
        Maximum progress bar value, by default None.
    epoch : Optional[int], optional
        Zero-indexed current epoch, by default None.
    num_epochs : Optional[int], optional
        Total number of epochs, by default None.
    stateful_metrics : Optional[list], optional
        Iterable of string names of metrics that should *not* be averaged over time.
        Metrics in this list will be displayed as-is. All others will be averaged by
        the progress bar before display, by default None.
    always_stateful : bool, optional
            Whether to set all metrics to be stateful, by default False.
    mode : str, optional
        Mode, one of "train", "val", or "predict", by default "train".
    """

    def __init__(
        self,
        max_value: Optional[int] = None,
        epoch: Optional[int] = None,
        num_epochs: Optional[int] = None,
        stateful_metrics: Optional[list] = None,
        always_stateful: bool = False,
        mode: str = "train",
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        max_value : Optional[int], optional
            Maximum progress bar value, by default None.
        epoch : Optional[int], optional
            Zero-indexed current epoch, by default None.
        num_epochs : Optional[int], optional
            Total number of epochs, by default None.
        stateful_metrics : Optional[list], optional
            Iterable of string names of metrics that should *not* be averaged over time.
            Metrics in this list will be displayed as-is. All others will be averaged by
            the progress bar before display, by default None.
        always_stateful : bool, optional
             Whether to set all metrics to be stateful, by default False.
        mode : str, optional
            Mode, one of "train", "val", or "predict", by default "train".
        """
        self.max_value = max_value
        # Width of the progress bar
        self.width = 30
        self.always_stateful = always_stateful

        if (epoch is not None) and (num_epochs is not None):
            print(f"Epoch: {epoch + 1}/{num_epochs}")

        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values: dict[Any, Any] = {}
        self._values_order: list[Any] = []
        self._start = time.time()
        self._last_update = 0.0
        self.spin = self.spinning_cursor() if self.max_value is None else None
        if mode == "train" and self.max_value is None:
            self.message = "Estimating dataset size"
        elif mode == "val":
            self.message = "Validating"
        elif mode == "predict":
            self.message = "Denoising"

    def update(
        self, current_step: int, batch_size: int = 1, values: Optional[list] = None
    ) -> None:
        """
        Update the progress bar.

        Parameters
        ----------
        current_step : int
            Index of the current step.
        batch_size : int, optional
            Batch size, by default 1.
        values : Optional[list], optional
            Updated metrics values, by default None.
        """
        values = values or []
        for k, v in values:
            # if torch tensor, convert it to numpy
            if str(type(v)) == "<class 'torch.Tensor'>":
                v = v.detach().cpu().numpy()

            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics and not self.always_stateful:
                if k not in self._values:
                    self._values[k] = [
                        v * (current_step - self._seen_so_far),
                        current_step - self._seen_so_far,
                    ]
                else:
                    self._values[k][0] += v * (current_step - self._seen_so_far)
                    self._values[k][1] += current_step - self._seen_so_far
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]

        self._seen_so_far = current_step

        now = time.time()
        info = f" - {(now - self._start):.0f}s"

        prev_total_width = self._total_width
        if self._dynamic_display:
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")
        else:
            sys.stdout.write("\n")

        if self.max_value is not None:
            bar = f"{current_step}/{self.max_value} ["
            progress = float(current_step) / self.max_value
            progress_width = int(self.width * progress)
            if progress_width > 0:
                bar += "=" * (progress_width - 1)
                if current_step < self.max_value:
                    bar += ">"
                else:
                    bar += "="
            bar += "." * (self.width - progress_width)
            bar += "]"
        else:
            bar = (
                f"{self.message} {next(self.spin)}, tile "  # type: ignore
                f"No. {current_step * batch_size}"
            )

        self._total_width = len(bar)
        sys.stdout.write(bar)

        if current_step > 0:
            time_per_unit = (now - self._start) / current_step
        else:
            time_per_unit = 0

        if time_per_unit >= 1 or time_per_unit == 0:
            info += f" {time_per_unit:.0f}s/step"
        elif time_per_unit >= 1e-3:
            info += f" {time_per_unit * 1e3:.0f}ms/step"
        else:
            info += f" {time_per_unit * 1e6:.0f}us/step"

        for k in self._values_order:
            info += f" - {k}:"
            if isinstance(self._values[k], list):
                avg = self._values[k][0] / max(1, self._values[k][1])
                if abs(avg) > 1e-3:
                    info += f" {avg:.4f}"
                else:
                    info += f" {avg:.4e}"
            else:
                info += f" {self._values[k]}s"

        self._total_width += len(info)
        if prev_total_width > self._total_width:
            info += " " * (prev_total_width - self._total_width)

        if self.max_value is not None and current_step >= self.max_value:
            info += "\n"

        sys.stdout.write(info)
        sys.stdout.flush()

        self._last_update = now

    def add(self, n: int, values: Optional[list] = None) -> None:
        """
        Update the progress bar by n steps.

        Parameters
        ----------
        n : int
            Number of steps to increase the progress bar with.
        values : Optional[list], optional
            Updated metrics values, by default None.
        """
        self.update(self._seen_so_far + n, 1, values=values)

    def spinning_cursor(self) -> Generator:
        """
        Generate a spinning cursor animation.

        Taken from https://github.com/manrajgrover/py-spinners/tree/master.

        Returns
        -------
        Generator
            Generator of animation frames.
        """
        while True:
            yield from [
                "▓ ----- ▒",
                "▓ ----- ▒",
                "▓ ----- ▒",
                "▓ ->--- ▒",
                "▓ ->--- ▒",
                "▓ ->--- ▒",
                "▓ -->-- ▒",
                "▓ -->-- ▒",
                "▓ -->-- ▒",
                "▓ --->- ▒",
                "▓ --->- ▒",
                "▓ --->- ▒",
                "▓ ----> ▒",
                "▓ ----> ▒",
                "▓ ----> ▒",
                "▒ ----- ░",
                "▒ ----- ░",
                "▒ ----- ░",
                "▒ ->--- ░",
                "▒ ->--- ░",
                "▒ ->--- ░",
                "▒ -->-- ░",
                "▒ -->-- ░",
                "▒ -->-- ░",
                "▒ --->- ░",
                "▒ --->- ░",
                "▒ --->- ░",
                "▒ ----> ░",
                "▒ ----> ░",
                "▒ ----> ░",
            ]
