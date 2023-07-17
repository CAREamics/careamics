from pathlib import Path
from typing import Dict

import torch

import wandb
from careamics_restoration.config import Configuration
from wandb import Settings


class WandBLogging:
    def __init__(
        self,
        experiment_name: str,
        log_path: Path,
        config: Configuration = None,
        model_to_watch: torch.nn.Module = None,
        save_code: bool = True
    ):
        self.run = wandb.init(
            dir=log_path,
            name=experiment_name,
            settings=Settings(silent="true", console="off"),
            config=config.model_dump() if config else None,
            save_code=save_code,
        )
        if model_to_watch:
            wandb.watch(model_to_watch, log="all", log_freq=1)

    def log_metrics(self, metric_dict: Dict):
        self.run.log(metric_dict, commit=True)
