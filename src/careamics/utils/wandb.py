"""
A WandB logger for CAREamics.

Implements a WandB class for use within the Engine.
"""
import sys
from pathlib import Path
from typing import Dict, Union

import torch
import wandb

from ..config import Configuration


def is_notebook() -> bool:
    """
    Check if the code is executed from a notebook or a qtconsole.

    Returns
    -------
    bool
        True if the code is executed from a notebooks, False otherwise.
    """
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
    """
    WandB logging class.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    log_path : Path
        Path in which to save the WandB log.
    config : Configuration
        Configuration of the model.
    model_to_watch : torch.nn.Module
        Model.
    save_code : bool, optional
        Whether to save the code, by default True.
    """

    def __init__(
        self,
        experiment_name: str,
        log_path: Path,
        config: Configuration,
        model_to_watch: torch.nn.Module,
        save_code: bool = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        experiment_name : str
            Name of the experiment.
        log_path : Path
            Path in which to save the WandB log.
        config : Configuration
            Configuration of the model.
        model_to_watch : torch.nn.Module
            Model.
        save_code : bool, optional
            Whether to save the code, by default True.
        """
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
                code_path = Path("../")
            self.log_code(code_path)

    def log_metrics(self, metric_dict: Dict) -> None:
        """
        Log metrics to wandb.

        Parameters
        ----------
        metric_dict : Dict
            New metrics entry.
        """
        self.run.log(metric_dict, commit=True)

    def log_code(self, code_path: Union[str, Path]) -> None:
        """
        Log code to wandb.

        Parameters
        ----------
        code_path : Union[str, Path]
            Path to the code.
        """
        self.run.log_code(
            root=code_path,
            include_fn=lambda path: path.endswith(".py")
            or path.endswith(".yml")
            or path.endswith(".yaml"),
        )
