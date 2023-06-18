from enum import Enum
from pathlib import Path
from typing import Union, Optional

from pydantic import BaseModel, validator

from .algorithm import Algorithm
from .training import Training
from .evaluation import Evaluation
from .prediction import Prediction


class Stage(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


class Configuration(BaseModel):
    """Main experiment configuration.

    Attributes
    ----------
    experiment_name : str
        Name of the experiment
    workdir : Path
        Path to the working directory
    algorithm : Algorithm
        Algorithm configuration
    training : Training
        Training configuration (optional)
    evaluation : Evaluation
        Evaluation configuration (optional)
    prediction : Prediction
        Prediction configuration (optional)
    """

    experiment_name: str
    workdir: Path

    # sub-configuration
    algorithm: Algorithm

    # other parameters are optional
    # these default to none and are omitted from yml export if not set
    training: Optional[Training] = None
    evaluation: Optional[Evaluation] = None
    prediction: Optional[Prediction] = None

    @validator("workdir")
    def validate_workdir(cls, v: Union[Path, str], **kwargs) -> Path:
        """Validate workdir.

        Parameters
        ----------
        v : Union[Path, str]
            Value to validate

        Returns
        -------
        Path
            Validated value

        Raises
        ------
        ValueError
            If workdir does not exist
        """
        path = Path(v)
        if not path.exists():
            raise ValueError(f"workdir {path} does not exist")

        return path

    def get_stage_config(self, stage: Union[str, Stage]) -> Union[Training, Evaluation]:
        """Get the configuration for a specific stage (training, evaluation or
        prediction).

        Parameters
        ----------
        stage : Union[str, Stage]
            Configuration stage: training, evaluation or prediction

        Returns
        -------
        Union[Training, Evaluation]
            Configuration for the specified stage

        Raises
        ------
        ValueError
            If stage is not one of training, evaluation or prediction
        """
        if stage == Stage.TRAINING:
            if self.training is None:
                raise ValueError("Training configuration is not defined.")

            return self.training
        elif stage == Stage.EVALUATION:
            if self.evaluation is None:
                raise ValueError("Evaluation configuration is not defined.")

            return self.evaluation
        elif stage == Stage.PREDICTION:
            if self.prediction is None:
                raise ValueError("Prediction configuration is not defined.")

            return self.prediction
        else:
            raise ValueError(
                f"Unknown stage {stage}. Available stages are"
                f"{Stage.TRAINING}, {Stage.EVALUATION} and"
                f"{Stage.PREDICTION}."
            )

    def dict(self) -> dict:
        """Override dict method.

        The purpose is to ensure export smooth import to yaml. It includes:
            - remove entries with None value
            - replace Path by str
        """
        dictionary = super().dict(exclude_none=True)

        # replace Path by str
        dictionary["workdir"] = str(dictionary["workdir"])

        return dictionary


def load_configuration(cfg_path: Union[str, Path]) -> dict:
    # TODO: import here because it might not be used everytime?
    # e.g. when using a library of config
    import yaml
    import re

    """Load a yaml config file and correct all datatypes."""
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    return yaml.load(Path(cfg_path).open("r"), Loader=loader)


def save_configuration(config: Configuration, path: Union[str, Path]) -> Path:
    """Save a configuration to a yaml file.

    Parameters
    ----------
    config : Configuration
        Configuration to save
    path : Union[str, Path]
        Path to the yaml file
    """
    import yaml

    if path.is_dir():
        path = Path(path, "config.yml")
    elif path.is_file() and path.suffix != ".yml":
        raise ValueError(f"Path must be a directory or .yml file (got {path}).")

    with open(path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)

    return path
