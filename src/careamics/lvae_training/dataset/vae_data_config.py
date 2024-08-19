from typing import Any, Optional
from enum import Enum

from pydantic import BaseModel, ConfigDict, computed_field


class DataType(Enum):
    MNIST = 0
    Places365 = 1
    NotMNIST = 2
    OptiMEM100_014 = 3
    CustomSinosoid = 4
    Prevedel_EMBL = 5
    AllenCellMito = 6
    SeparateTiffData = 7
    CustomSinosoidThreeCurve = 8
    SemiSupBloodVesselsEMBL = 9
    Pavia2 = 10
    Pavia2VanillaSplitting = 11
    ExpansionMicroscopyMitoTub = 12
    ShroffMitoEr = 13
    HTIba1Ki67 = 14
    BSD68 = 15
    BioSR_MRC = 16
    TavernaSox2Golgi = 17
    Dao3Channel = 18
    ExpMicroscopyV2 = 19
    Dao3ChannelWithInput = 20
    TavernaSox2GolgiV2 = 21
    TwoDset = 22
    PredictedTiffData = 23
    Pavia3SeqData = 24
    # Here, we have 16 splitting tasks.
    NicolaData = 25


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


class GridAlignement(Enum):
    """
    A patch is formed by padding the grid with content. If the grids are 'Center' aligned, then padding is to done equally on all 4 sides.
    On the other hand, if grids are 'LeftTop' aligned, padding is to be done on the right and bottom end of the grid.
    In the former case, one needs (patch_size - grid_size)//2 amount of content on the right end of the frame.
    In the latter case, one needs patch_size - grid_size amount of content on the right end of the frame.
    """

    LeftTop = 0
    Center = 1


# TODO: for all bool params check if they are taking different values in Disentangle repo
# TODO: check if any bool logic can be removed
class VaeDatasetConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    data_type: Optional[DataType]
    multiscale_lowres_count: Optional[int] = None

    depth3D: Optional[int] = 1

    datasplit_type: Optional[DataSplitType] = None

    num_channels: Optional[int] = 2
    ch1_fname: Optional[str] = None
    ch2_fname: Optional[str] = None
    ch_input_fname: Optional[str] = None

    input_is_sum: Optional[bool] = False
    input_idx: Optional[int] = None
    target_idx_list: Optional[Any] = None

    start_alpha: Optional[Any] = None
    end_alpha: Optional[Any] = None
    alpha_weighted_target: Optional[bool] = False

    image_size: int
    grid_size: Optional[int] = None

    std_background_arr: Optional[Any] = None

    empty_patch_replacement_enabled: Optional[bool] = False
    empty_patch_replacement_channel_idx: Optional[Any] = None
    empty_patch_replacement_probab: Optional[Any] = None
    empty_patch_max_val_threshold: Optional[Any] = None

    uncorrelated_channels: Optional[bool] = False
    uncorrelated_channel_probab: Optional[float] = 0.5

    poisson_noise_factor: Optional[float] = -1
    synthetic_gaussian_scale: Optional[float] = 0.1
    input_has_dependant_noise: Optional[bool] = False
    enable_gaussian_noise: Optional[bool] = False
    allow_generation: bool = False

    # Not used
    training_validtarget_fraction: Any = None
    deterministic_grid: Any = None

    enable_rotation_aug: Optional[bool] = False

    grid_alignment: GridAlignement = GridAlignement.LeftTop

    max_val: Optional[float] = None
    trim_boundary: Optional[bool] = True

    overlapping_padding_kwargs: Any = None
    print_vars: Optional[bool] = False

    # Hard-coded parameters (used to be in the config file)
    normalized_input: bool = True
    """If this is set to true, then one mean and stdev is used
                for both channels. Otherwise, two different mean and stdev are used."""
    use_one_mu_std: Optional[bool] = True
    train_aug_rotate: Optional[bool] = False
    enable_random_cropping: Optional[bool] = True
    lowres_supervision: Optional[bool] = False

    @computed_field
    @property
    def padding_kwargs(self) -> dict:
        kwargs_dict = {}
        padding_kwargs = {}
        if (
            self.multiscale_lowres_count is not None
            and self.multiscale_lowres_count is not None
        ):
            # Get padding attributes
            if "padding_kwargs" not in kwargs_dict:
                padding_kwargs = {}
                padding_kwargs["mode"] = "reflect"
                padding_kwargs["constant_values"] = None
            else:
                padding_kwargs = kwargs_dict.pop("padding_kwargs")
        return padding_kwargs

    # # from Disentangle biosr config
    # sampler_type: Any
    # threshold: Any
    # normalized_input: Any
    # clip_percentile: Any
    # channelwise_quantile: Any
    # use_one_mu_std: Any
    # train_aug_rotate: Any
    # randomized_channels: Any
    # padding_mode: Any
    # padding_value: Any
    # target_separate_normalization: Any
