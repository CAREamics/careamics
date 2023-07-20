from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from wandb import Settings

from careamics_restoration.config import Configuration


class WandBLogging:
    def __init__(
        self,
        experiment_name: str,
        log_path: Path,
        # TODO shouldn't it be always not None?
        config: Optional[Configuration] = None,
        # TODO same here
        model_to_watch: Optional[torch.nn.Module] = None,
        save_code: bool = True,
    ):
        self.run = wandb.init(
            dir=log_path,
            name=experiment_name,
            settings=Settings(silent="true", console="off"),
            # TODO should we dump the whole configuration (including default optionals)?
            config=config.model_dump() if config else None,
            save_code=save_code,
        )
        if model_to_watch:
            wandb.watch(model_to_watch, log="all", log_freq=1)

    def log_metrics(self, metric_dict: Dict):
        self.run.log(metric_dict, commit=True)
