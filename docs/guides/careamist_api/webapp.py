"""Next-Gen CAREamics example for the webapp.

Note that this example covers only the webapp parameters and the configuration.
"""

from pathlib import Path

from pydantic import BaseModel, field_validator, Field

from careamics.config.validators import check_axes_validity
from careamics.config.ng_factories import create_advanced_n2v_config

InputData = str | Path | list[str] | list[Path]


# TODO defaults will need to be adjusted
class WebAppConfigTraining(BaseModel):
    """Configuration for the webapp."""

    # data-related parameters, to be passed to train function
    train_images: InputData
    """Input data for training, can be a single file or multiple files. Images must be
    in tiff format (*.tif*)."""

    val_images: InputData | None = None
    """Input data for validation, can be a single file or multiple files. Images must be
    in tiff format (*.tif*). If not provided, validation will be taken from training
    images."""

    # configuration related parameters
    experiment_name: str
    """Name of the experiment, used for logging and saving the model."""

    enable_3d: bool = False
    """Whether to use 3D."""

    axes: str
    """Axes of the data. Must be a combination of 'STCZYX', must contain at least X and
    Y axes, X and Y must be contiguous, and axes must not contain duplicates."""

    num_epochs: int = Field(default=30, ge=1, le=300)
    """Number of epochs to train the model for."""

    num_steps: int | None = Field(default=None, ge=1, le=1000)
    """Number of steps to train the model for. If not provided, it will be inferred from
    the size of the data."""

    batch_size: int = Field(default=32, ge=1, le=256)
    """Number of samples per batch."""

    patch_xy: int = Field(default=64, ge=16, le=512)
    """Size of the patches to be extracted from the images for training."""

    patch_z: int | None = Field(default=None, ge=1, le=64)
    """Size of the patches to be extracted from the images for training. Only used if
    enable_2d is False."""

    n_channels: int = Field(default=1, ge=1)
    """Number of channels in the input data."""

    # advanced parameters
    # Note: this parameter is currently not implemented but will be soon
    n_val_patches: int = Field(default=20, ge=1, le=100)
    """Number of patches to be extracted from the training images for validation. Only
    used if val_images is not provided."""

    x_flip: bool = True
    """Whether to apply horizontal flip as data augmentation."""

    y_flip: bool = True
    """Whether to apply vertical flip as data augmentation."""

    rot90: bool = True
    """Whether to apply 90 degree rotation as data augmentation."""

    use_n2v2: bool = False
    """Whether to use the N2V2 architecture."""

    depth: int = Field(default=3, ge=2, le=5)
    """Depth of the U-Net architecture."""

    n_filters: int = Field(default=32, ge=16, le=128)
    """Number of filters in the first layer of the U-Net architecture."""

    independent_channels: bool = True
    """Whether to treat channels independently."""

    @field_validator("axes")
    def validate_axes(cls, v: str) -> str:
        """Validate the axes string."""
        # Note: in the future this will become a proper validator, we just haven't done
        # that yet (f(str)->str)
        check_axes_validity(v)
        return v

    def get_careamics_config_dict(self) -> dict:
        """Get a dictionary of parameters to be passed to the configuration."""
        # create augmentations
        augs = []
        if self.x_flip:
            augs.append("x_flip")
        if self.y_flip:
            augs.append("y_flip")
        if self.rot90:
            augs.append("rot90")

        config_dict = {
            "experiment_name": self.experiment_name,
            "data_type": "tiff",
            "axes": self.axes,
            "patch_size": (
                [self.patch_z, self.patch_xy, self.patch_xy]
                if self.enable_3d
                else [self.patch_xy, self.patch_xy]
            ),
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "num_steps": self.num_steps,
            "n_channels": self.n_channels,
            "augmentations": augs,
            "independent_channels": self.independent_channels,
            "use_n2v2": self.use_n2v2,
            "num_workers": 8,  # TODO will need to be benchmarked
            "seed": 42,
        }
        return config_dict


def main():
    """Main function to run the webapp."""
    webapp_config = WebAppConfigTraining(
        train_images="path/to/train/images",
        val_images="path/to/val/images",
        experiment_name="my_experiment",
        enable_3d=True,
        axes="CZYX",
        num_epochs=50,
        num_steps=100,
        batch_size=16,
        patch_xy=128,
        patch_z=16,
        n_channels=3,
        n_val_patches=50,
        x_flip=True,
        y_flip=True,
        rot90=False,
        use_n2v2=True,
        depth=4,
        n_filters=64,
        independent_channels=False,
    )

    # TODO this will raise errors
    careamics_config = create_advanced_n2v_config(
        **webapp_config.get_careamics_config_dict()
    )


if __name__ == "__main__":
    main()
