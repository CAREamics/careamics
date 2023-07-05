import logging
import sys

from rich.console import Group, Console
from rich.logging import RichHandler
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.live import Live
from rich_pixels import Pixels


banner = """                                                                                                                         
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
loggers = {}


def get_logger(name, log_level=logging.INFO):
    logger = logging.getLogger(name)
    if name in loggers:
        return logger

    for logger_name in loggers:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    handlers = [RichHandler(rich_tracebacks=True, show_level=False)]

    formatter = logging.Formatter("%(message)s")

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    loggers[name] = True
    return logger


class ProgressLogger:
    def __init__(self):
        self.in_notebook = 'ipykernel' in sys.modules

        self.console = Console()

        self.tasks = {}
        self.total_progress = Progress(
            TextColumn("{task.description}", justify="right"),
            MofNCompleteColumn(),
            BarColumn(bar_width=60),
            TextColumn("Time elapsed: "),
            TimeElapsedColumn(),
            TextColumn("Time remaining: "),
            TimeRemainingColumn(compact=True)
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
            Padding(self.total_progress, (1, 8)),
            Padding(self.task_progress, (0, 5))
        )

        if not self.in_notebook:
            pixels = Pixels.from_ascii(banner)
            header_panel = Panel.fit(pixels, style="red", padding=1)
            self.interface = Group(
                header_panel, progress_group
            )
        else:
            self.interface = Group(
                progress_group
            )
        self.live = None

    def _get_task(self, task_name, task_length, tracker):
        if task_name not in self.tasks:
            task_id = tracker.add_task(task_name, total=task_length)
            self.tasks[task_name] = task_id
        else:
            task_id = self.tasks[task_name]
            tracker.reset(task_id, visible=True)
            tracker.start_task(task_id)

        return task_id

    def _start_live_if_needed(self):
        if not self.live:
            self.live = Live(self.interface)
            self.live.__enter__()

    def exit(self):
        self.live.__exit__(None, None, None)
        self.live = None

    def __call__(
        self, task_iterable, task_name, overall_progress=False, persistent=True, endless=False
    ):
        self._start_live_if_needed()

        if overall_progress:
            tracker = self.total_progress
        else:
            tracker = self.task_progress

        if endless:
            task_length = None
        else:
            task_length = len(task_iterable)
        task_id = self._get_task(task_name, task_length, tracker=tracker)

        for item in task_iterable:
            yield item
            tracker.update(task_id, advance=1)

        if not persistent:
            tracker.reset(task_id, visible=False)
            self.tasks.pop(task_name)
