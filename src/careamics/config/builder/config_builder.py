from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypedDict, TypeVar

from careamics.config.algorithms import CAREAlgorithm, N2NAlgorithm, N2VAlgorithm
from careamics.config.data.data_config import _is_3D
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


class ConfigDict(TypedDict):
    experiment_name: str
    algorithm_config: dict[str, Any]
    data_config: dict[str, Any]
    training_config: dict[str, Any]


class ConfigBuilder(Protocol):
    # mutable ref of config dict
    config_dict: ConfigDict

    # basic args
    data_type: Literal["array", "tiff", "zarr", "czi", "custom"]
    axes: str
    patch_size: Sequence[int]
    batch_size: int
    num_epochs: int
    num_steps: int | None
    n_channels_in: int | None
    n_channels_out: int | None
    seed: int | None

    @property
    def is_3D(self) -> bool: ...

    def build(self) -> Configuration | N2VConfiguration: ...


class BaseConfigBuilder(ConfigBuilder):
    def __init__(
        self,
        experiment_name: str,
        data_type: Literal["array", "tiff", "zarr", "czi", "custom"],
        axes: str,
        patch_size: Sequence[int],
        batch_size: int,
        # optional
        num_epochs: int = 30,
        num_steps: int | None = None,
        n_channels_in: int | None = None,
        n_channels_out: int | None = None,
        seed: int | None = None,
    ):
        self.config_dict = {
            "experiment_name": experiment_name,
            "algorithm_config": {},
            "data_config": {},
            "training_config": {},
        }
        self.data_type = data_type
        self.axes = axes
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.seed = seed
        super().__init__()

    @property
    def is_3D(self) -> bool:
        return _is_3D(self.axes, SupportedData(self.data_type))

    def _before_build(self) -> None:
        """Hook"""
        pass

    def build(self) -> ConfigurationType:
        self._before_build()
        return instantiate_config(self.config_dict)
