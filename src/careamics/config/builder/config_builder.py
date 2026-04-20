from collections.abc import Sequence
from typing import Any, Literal, NotRequired, Protocol, TypeVar

from typing_extensions import TypedDict

from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from careamics.config.support import SupportedData

from ..configuration import Configuration
from ..factories.config_discriminators import instantiate_config
from ..n2v_configuration import N2VConfiguration

ConfigurationType = (
    Configuration[CAREAlgorithm]
    | Configuration[N2NAlgorithm]
    | Configuration[N2VAlgorithm]
)

ConfigBuilderT = TypeVar("ConfigBuilderT", bound="ConfigBuilder")


# type ignore extra_items, mypy doesn't see it is the TypedDict from typing_extensions
class DataConfigDict(TypedDict, extra_items=Any):  # type: ignore[call-arg]
    mode: str
    axes: str
    data_type: SupportedData
    patching: "PatchingConfigDict"
    batch_size: int


class PatchingConfigDict(TypedDict, extra_items=Any):  # type: ignore[call-arg]
    name: Literal["random", "stratified"]
    patch_size: Sequence[int]


class AlgorithmConfigDict(TypedDict, extra_items=Any):  # type: ignore[call-arg]
    algorithm: str
    model: dict[str, Any]


class TrainingConfigDict(TypedDict, extra_items=Any):  # type: ignore[call-arg]
    trainer_params: NotRequired[dict[str, Any]]
    checkpoint_params: NotRequired[dict[str, Any]]
    early_stopping_params: NotRequired[dict[str, Any] | None]


class ConfigDict(TypedDict):
    experiment_name: str
    algorithm_config: AlgorithmConfigDict
    data_config: DataConfigDict
    training_config: TrainingConfigDict


class ConfigBuilder(Protocol):
    # mutable ref of config dict
    config_dict: ConfigDict

    seed: int | None

    def build(self) -> Configuration | N2VConfiguration: ...


class BaseConfigBuilder(ConfigBuilder):
    # @property
    # def is_3D(self) -> bool:
    #     return _is_3D(
    #         self.config_dict["data_config"]["axes"],
    #         self.config_dict["data_config"]["data_type"],
    #     )

    def _before_build(self) -> None:
        """Hook"""
        pass

    def build(self) -> ConfigurationType:
        self._before_build()
        return instantiate_config(self.config_dict)
