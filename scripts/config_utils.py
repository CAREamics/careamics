import pickle

from careamics.config.data.microsplit_data_config import MicroSplitDataConfig
from careamics.config.data.normalization_config import MeanStdConfig
from careamics.config.data.data_config import (
    SlidingWindowTiledPatchingConfig,
    TiledPatchingConfig,
)


def pkl_load(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def get_predict_config(
    pkl_data: dict,
    *,
    overlap_size: list[int],
    stride: list[int] | None = None,
    image_means: list[float],
    image_stds: list[float],
    target_means: list[float],
    target_stds: list[float],
    batch_size: int = 1,
) -> "MicroSplitDataConfig":
    is_3d = pkl_data.get("mode_3D", False)
    axes = "SCZYX" if is_3d else "SCYX"
    img = pkl_data["image_size"]
    patch_size = [pkl_data["depth3D"], img, img] if is_3d else [img, img]

    patching = (
        SlidingWindowTiledPatchingConfig(
            patch_size=patch_size, overlaps=overlap_size, stride=stride
        )
        if stride is not None
        else TiledPatchingConfig(
            patch_size=patch_size, overlaps=overlap_size
        )
    )

    return MicroSplitDataConfig(
        mode="predicting",
        data_type="tiff",
        axes=axes,
        patching=patching,
        normalization=MeanStdConfig(
            image_means=image_means, 
            image_stds=image_stds,
            target_means=target_means,
            target_stds=target_stds,
        ),
        multiscale_count=pkl_data["multiscale_lowres_count"],
        padding_mode=pkl_data["padding_mode"],
        batch_size=batch_size,
        augmentations=[], # predict mode: no augs
    )
