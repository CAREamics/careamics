from rich.console import Group, Console
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


class ProgressLogger:
    def __init__(self):
        self.console = Console()

        self.tasks = {}
        self.total_progress = Progress(
            TextColumn("{task.description}", justify="right"),
            MofNCompleteColumn(),
            BarColumn(bar_width=60),
            TextColumn("Time elapsed: "),
            TimeElapsedColumn(),
        )
        self.task_progress = Progress(
            TextColumn("{task.description}", justify="right"),
            SpinnerColumn(spinner_name="point", style="red"),
            MofNCompleteColumn(),
            BarColumn(bar_width=60),
            TextColumn("Time remaining: "),
            TimeRemainingColumn(compact=True),
        )

        pixels = Pixels.from_ascii(banner)
        header_panel = Panel.fit(pixels, style="red", padding=1)
        self.console.print(header_panel)
        self.progress_group = Group(
            Padding(self.total_progress, (1, 13)), self.task_progress
        )
        self.interface = Panel(
            self.progress_group, style="red", padding=1, title="Progress"
        )

        self.live = Live(self.interface)
        self.live.__enter__()

    def _get_task(self, task_name, task_length, tracker):
        if task_name not in self.tasks:
            task_id = tracker.add_task(task_name, total=task_length, start=True)
            self.tasks[task_name] = task_id
        else:
            task_id = self.tasks[task_name]
            tracker.reset(task_id, visible=True)
            tracker.start_task(task_id)

        return task_id

    def __call__(
        self, task_iterable, logger_name, overall_progress=False, persistent=True
    ):
        if overall_progress:
            tracker = self.total_progress
        else:
            tracker = self.task_progress

        task_length = len(task_iterable)
        task_id = self._get_task(logger_name, task_length, tracker=tracker)

        for item in task_iterable:
            yield item
            tracker.update(task_id, advance=1)

        if not persistent:
            tracker.reset(task_id, visible=False)
            self.tasks.pop(logger_name)
