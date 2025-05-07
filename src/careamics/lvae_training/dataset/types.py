from enum import Enum


class DataType(Enum):
    HTH24Data = 0
    HTLIF24Data = 1
    PaviaP24Data = 2
    TavernaSox2GolgiV2 = 3
    Dao3ChannelWithInput = 4
    ExpMicroscopyV1 = 5
    ExpMicroscopyV2 = 6
    Dao3Channel = 7
    TavernaSox2Golgi = 8
    HTIba1Ki67 = 9
    OptiMEM100_014 = 10
    SeparateTiffData = 11
    BioSR_MRC = 12
    HTH23BData = 13  # puncta, in case we have differently sized crops for each channel.
    Care3D = 14


class DataSplitType(Enum):
    All = 0
    Train = 1
    Val = 2
    Test = 3


class TilingMode(Enum):
    TrimBoundary = 0
    PadBoundary = 1
    ShiftBoundary = 2
