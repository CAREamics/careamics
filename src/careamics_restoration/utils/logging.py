import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich_pixels import Pixels

banner: str = """
   ......       ......     ........     ........                                   ....
 -+++----+-   -+++--+++-  :+++---+++:  :+++-----                                   .--:
.+++     .:   +++.  .+++. :+++   :+++  :+++         :------.   .---:----..:----.   :---    :----:     :----:.
.+++         .+++.  .+++. :+++   -++=  :+++        +=....=+++  :+++-..=+++-..=++=  -+++  .+++-..++   +++-..=+.
.+++         .++++++++++. :++++++++=.  :++++++:          .+++. :+++   :+++   -+++  -+++  :+++       .+++=.
.+++         .+++.  .+++. :+++   -+++  :+++        :=++==++++. :+++   :+++   -+++  -+++  :+++        .-=+++=:
.+++     ..  .+++.  .+++. :+++   :+++  :+++       .+++.  .+++. :+++   :+++   -+++  -+++  :+++   ..   ..  :+++.
 -++=-::-+=  .+++.  .+++. :+++   :+++  :+++-::::   =++=--=+++. :+++   :+++   -+++  -+++   =++=:-+=   =+-:=++=
   ......     ...    ...   ...    ...   ........     .... ...   ...    ...   ....  ....     ....      .....
"""

LOGGERS: dict = {}


# TODO: export all the loggers to the same file
def get_logger(
    name: str, log_level=logging.INFO, log_path: Optional[Union[str, Path]] = None
):
    logger = logging.getLogger(name)
    if name in LOGGERS:
        return logger

    for logger_name in LOGGERS:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    handlers = [
        RichHandler(rich_tracebacks=True, show_level=False),
    ]
    if log_path:
        handlers.append(
            logging.FileHandler(log_path),
        )

    formatter = logging.Formatter("%(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    LOGGERS[name] = True
    return logger


class ProgressLogger:
    def __init__(self):
        self.is_in_notebook = "ipykernel" in sys.modules

        self.console = Console()

        self.tasks = {}
        self.total_progress = Progress(
            TextColumn("{task.description}", justify="right"),
            MofNCompleteColumn(),
            BarColumn(bar_width=60),
            TextColumn("Time elapsed: "),
            TimeElapsedColumn(),
            TextColumn("Time remaining: "),
            TimeRemainingColumn(compact=True),
        )

        self.task_progress = Progress(
            TextColumn("{task.description}", justify="right"),
            SpinnerColumn(spinner_name="point", style="red"),
            MofNCompleteColumn(),
            BarColumn(bar_width=60),
            TextColumn("Time elapsed: "),
            TimeElapsedColumn(),
        )

        progress_group = Group(
            Padding(self.total_progress, (1, 8)), Padding(self.task_progress, (0, 5))
        )

        if not self.is_in_notebook:
            pixels = Pixels.from_ascii(banner)
            header_panel = Panel.fit(pixels, style="red", padding=1)
            self.console.print(header_panel)

        self.interface = Group(progress_group)
        self.live = None

    def _start_live_if_needed(self):
        if not self.live:
            self.live = Live(self.interface)
            self.live.__enter__()

    def _get_task(self, task_name: str, task_length: Optional[int], tracker: Progress):
        if task_name not in self.tasks:
            task_id = tracker.add_task(task_name, total=task_length)
            self.tasks[task_name] = task_id
        else:
            task_id = self.tasks[task_name]
            tracker.reset(task_id, visible=True)
            tracker.start_task(task_id)

        return task_id

    def exit(self):
        self.live.__exit__(None, None, None)
        self.live = None

    def __call__(
        self,
        task_iterable: Iterable,
        task_name: str,
        overall_progress: bool = False,
        persistent: bool = True,
        unbounded: bool = False,
    ):
        self._start_live_if_needed()

        if overall_progress:
            tracker = self.total_progress
        else:
            tracker = self.task_progress

        if unbounded:
            task_length = None
        else:
            # TODO in Engine, task_iterable is an enumeration, but neither enumerate nor Iterable have __len__
            task_length = len(task_iterable)
        task_id = self._get_task(task_name, task_length, tracker=tracker)

        for item in task_iterable:
            yield item
            tracker.update(task_id, advance=1)

        if not persistent:
            tracker.reset(task_id, visible=False)
            self.tasks.pop(task_name)
