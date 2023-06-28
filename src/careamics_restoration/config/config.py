from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict

from pydantic import BaseModel, validator

from .algorithm import Algorithm
from .evaluation import Evaluation
from .prediction import Prediction
from .stage import Stage
from .training import Training


class RunParams(BaseModel):
    """Basic parameters or current run."""

    experiment_name: str
    workdir: str
    trained_model: Optional[str] = None

    @validator("workdir")
    def validate_workdir(cls, v: str, values, **kwargs) -> Path:
        """Validate trained_model.

        If trained_model is not None, it must be a valid path.
        """
        path = Path(v)
        if path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @validator("trained_model")
    def validate_trained_model(
        cls, v: Union[Path, None], values, **kwargs
    ) -> Union[None, Path]:
        """Validate trained_model.

        If trained_model is not None, it must be a valid path.
        """
        if v is not None:
            path = values["workdir"] / Path(v)
            if not path.exists():
                raise ValueError(f"Path to model does not exist (got {v}).")
            elif path.suffix != ".pth":
                raise ValueError(f"Path to model must be a .pth file (got {v}).")
            else:
                return path

        return None


class ConfigStageEnum(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"
    PREDICTION = "prediction"


# TODO Discuss the structure and logic of the configuration, and document every constraints decision


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

    run_params: RunParams

    # sub-configuration
    algorithm: Algorithm

    # other parameters are optional
    # these default to none and are omitted from yml export if not set
    training: Optional[Training] = None
    evaluation: Optional[Evaluation] = None
    prediction: Optional[Prediction] = None

    def get_stage_config(self, stage: Union[str, ConfigStageEnum]) -> Stage:
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
        if stage == ConfigStageEnum.TRAINING:
            if self.training is None:
                raise ValueError("Training configuration is not defined.")

            return self.training
        elif stage == ConfigStageEnum.EVALUATION:
            if self.evaluation is None:
                raise ValueError("Evaluation configuration is not defined.")

            return self.evaluation
        elif stage == ConfigStageEnum.PREDICTION:
            if self.prediction is None:
                raise ValueError("Prediction configuration is not defined.")

            return self.prediction
        else:
            raise ValueError(
                f"Unknown stage {stage}. Available stages are"
                f"{ConfigStageEnum.TRAINING}, {ConfigStageEnum.EVALUATION} and"
                f"{ConfigStageEnum.PREDICTION}."
            )


def load_configuration(cfg_path: Union[str, Path]) -> Dict:
    # TODO: import here because it might not be used everytime?
    # e.g. when using a library of config
    import re
    import yaml

    """Load a yaml config file and correct all datatypes."""
    # TODO Igor: move this functionality to a pydantic validator and remove due to a popular request
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

    # make sure path is a Path object
    config_path = Path(path)

    if config_path.is_dir():
        config_path = Path(config_path, "config.yml")
    elif config_path.is_file() and config_path.suffix != ".yml":
        raise ValueError(f"Path must be a directory or .yml file (got {config_path}).")

    with open(config_path, "w") as f:
        yaml.dump(config.dict(), f, default_flow_style=False)

    return config_path
