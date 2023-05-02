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
        Training configuration
    evaluation : Evaluation
        Evaluation configuration
    prediction : Prediction
        Prediction configuration
    """

    experiment_name: str
    workdir: Path

    algorithm: Algorithm

    # the rest is optional
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
            - replace Path by str
            - remove entries with None value
        """
        dictionary = super().dict()

        # replace Path by str
        dictionary["workdir"] = str(dictionary["workdir"])

        # remove None values
        for key, value in list(dictionary.items()):
            if value is None:
                del dictionary[key]

        return dictionary
