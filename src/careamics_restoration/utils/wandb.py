import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb

from careamics_restoration.config import Configuration


def is_notebook() -> bool:
    """Check if the code is exectuted from a notebook."""
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        else:
            return False
    except (NameError, ModuleNotFoundError):
        return False


class WandBLogging:
    """WandB logging class."""

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
            project="careamics-restoration",
            dir=log_path,
            name=experiment_name,
            config=config.model_dump() if config else None,
            # save_code=save_code,
        )
        if model_to_watch:
            wandb.watch(model_to_watch, log="all", log_freq=1)
        if save_code:
            if is_notebook():
                # Get all sys path and select the root
                code_path = Path([p for p in sys.path if "caremics" in p][-1]).parent
            else:
                code_path = "../"
            self.log_code(code_path)

    def log_metrics(self, metric_dict: Dict) -> None:
        """Log metrics to wandb."""
        self.run.log(metric_dict, commit=True)

    def log_code(self, code_path: str) -> None:
        """Log code to wandb."""
        self.run.log_code(
            root=code_path,
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".yml")
            or path.endswith(".yaml"),
        )
