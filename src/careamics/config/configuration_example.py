"""Example of configurations."""

from .algorithm_model import AlgorithmConfig
from .architectures import UNetModel
from .configuration_model import Configuration
from .data_model import DataConfig
from .optimizer_models import LrSchedulerModel, OptimizerModel
from .support import (
    SupportedActivation,
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedData,
    SupportedLogger,
    SupportedLoss,
    SupportedOptimizer,
    SupportedPixelManipulation,
    SupportedScheduler,
    SupportedTransform,
)
from .training_model import TrainingConfig


def full_configuration_example() -> Configuration:
    """Return a dictionnary representing a full configuration example.

    Returns
    -------
    Configuration
        Full configuration example.
    """
    experiment_name = "Full example"
    algorithm_model = AlgorithmConfig(
        algorithm=SupportedAlgorithm.N2V.value,
        loss=SupportedLoss.N2V.value,
        model=UNetModel(
            architecture=SupportedArchitecture.UNET.value,
            in_channels=1,
            num_classes=1,
            depth=2,
            num_channels_init=32,
            final_activation=SupportedActivation.NONE.value,
            n2v2=True,
        ),
        optimizer=OptimizerModel(
            name=SupportedOptimizer.ADAM.value, parameters={"lr": 0.0001}
        ),
        lr_scheduler=LrSchedulerModel(
            name=SupportedScheduler.REDUCE_LR_ON_PLATEAU.value,
        ),
    )
    data_model = DataConfig(
        data_type=SupportedData.ARRAY.value,
        patch_size=(256, 256),
        batch_size=8,
        axes="YX",
        transforms=[
            {
                "name": SupportedTransform.XY_FLIP.value,
            },
            {
                "name": SupportedTransform.XY_RANDOM_ROTATE90.value,
            },
            {
                "name": SupportedTransform.N2V_MANIPULATE.value,
                "roi_size": 11,
                "masked_pixel_percentage": 0.2,
                "strategy": SupportedPixelManipulation.MEDIAN.value,
            },
        ],
        mean=0.485,
        std=0.229,
        dataloader_params={
            "num_workers": 4,
        },
    )
    training_model = TrainingConfig(
        num_epochs=30,
        logger=SupportedLogger.WANDB.value,
    )

    return Configuration(
        experiment_name=experiment_name,
        algorithm_config=algorithm_model,
        data_config=data_model,
        training_config=training_model,
    )
