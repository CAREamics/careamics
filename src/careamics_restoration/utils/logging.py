import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

LOGGERS: dict = {}


def get_logger(
    name: str,
    log_level: int = logging.INFO,
    log_path: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Creates a python logger instance with configured handlers."""
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
    """Keras style progress bar.

    Modified from https://github.com/yueyericardo/pkbar

    Arguments:
        max_value: Number of steps expected, None if unknown.
        epoch: Zero-indexed current epoch.
        num_epochs: Total epochs.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        always_stateful: (Boolean) Whether to set all metrics to be stateful.
        stateful_metrics: Iterable of string names of metrics that
                should *not* be averaged over time. Metrics in this list
                will be displayed as-is. All others will be averaged
                by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        max_value: Optional[int] = None,
        epoch: Optional[int] = None,
        num_epochs: Optional[int] = None,
        stateful_metrics: Optional[List] = None,
        always_stateful: bool = False,
        mode: str = "train",
    ) -> None:
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
        self._values: Dict[Any, Any] = {}
        self._values_order: List[Any] = []
        self._start = time.time()
        self._last_update = 0.0
        self.spin = self.spinning_cursor() if self.max_value is None else None
        if mode == "train":
            self.message = "Estimating dataset size"
        elif mode == "val":
            self.message = "Validating"
        elif mode == "predict":
            self.message = "Denoising"
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def update(
        self, current_step: int, batch_size: int = 1, values: Optional[List] = None
    ) -> None:
        """Updates the progress bar.

        Arguments:
                current_step: Index of current step.
                values: List of tuples:
                        `(name, value_for_last_step)`.
                        If `name` is in `stateful_metrics`,
                        `value_for_last_step` will be displayed as-is.
                        Else, an average of the metric over time will be displayed.
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
                avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
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

    def add(self, n: int, values: Optional[List] = None) -> None:
        """Adds progress."""
        self.update(self._seen_so_far + n, 1, values=values)

    def spinning_cursor(self) -> Generator:
        """Generates a spinning cursor animation.

        Taken from https://github.com/manrajgrover/py-spinners/tree/master
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
