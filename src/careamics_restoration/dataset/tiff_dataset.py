import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Dict

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from careamics_restoration.config import ConfigStageEnum
from careamics_restoration.dataset.tiling import (
    extract_patches_predict,
    extract_patches_sequential,
    extract_patches_random,
)
from careamics_restoration.manipulation import default_manipulate
from careamics_restoration.utils import normalize
from careamics_restoration.config.training import ExtractionStrategies

logger = logging.getLogger(__name__)


class TiffDataset(torch.utils.data.IterableDataset):
    """Dataset to extract patches from a list of images and apply transforms to the patches."""

    def __init__(
        self,
        data_path: Union[Path, str],
        axes: str,
        patch_extraction_method: str,
        patch_size: Union[List[int], Tuple[int]] = None,
        patch_overlap: Union[List[int], Tuple[int]] = None,
        mean: float = None,
        std: float = None,
        patch_transform: Optional[Callable] = None,
        patch_transform_params: Optional[Dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        data_path : str
            Path to data, must be a directory.
        axes: str
            Description of axes in format STCZYX
        patch_extraction_method: str
            aaa
        patch_size : Tuple[int]
            The size of the patch to extract from the image. Must be a tuple of len either 2 or 3
            depending on number of spatial dimension in the data.
        patch_overlap:
            aaa
        mean:
            aaa
        std:
            aaa
        image_transform : Optional[Callable], optional
            _description_, by default None
        patch_transform: Optional[Callable], optional
            aaa
        """
        self.data_path = data_path
        self.axes = axes

        self.patch_transform = patch_transform

        self.files = self.list_files()

        if not mean or not std:
            self.mean, self.std = self.calculate_mean_and_std()

        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.patch_extraction_method = patch_extraction_method
        self.patch_transform = patch_transform
        self.patch_transform_params = patch_transform_params

    def list_files(self) -> List[Path]:
        files = sorted(Path(self.data_path).rglob(f"*.tif*"))
        if len(files) == 0:
            raise ValueError(f"No tiff files found in {self.data_path}.")
        return files

    def calculate_mean_and_std(self) -> Tuple[float, float]:
        means, stds = 0, 0
        num_samples = 0

        for sample in tqdm(self.iterate_files()):
            means += sample.mean()
            stds += np.std(sample)
            num_samples += 1

        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")

        return result_mean, result_std

    # TODO add axes shuffling and reshapes. so far assuming correct order
    # TODO rewrite it to make it readable
    def fix_axes(self, sample):
        # concatenate ST axes to N, return NCZYX
        if ("S" in self.axes or "T" in self.axes) and sample.dtype != "O":
            new_axes_len = len(self.axes.replace("Z", "").replace("YX", ""))
            sample = sample.reshape(-1, *sample.shape[new_axes_len:])

        elif sample.dtype == "O":
            for i in range(len(sample)):
                sample[i] = np.expand_dims(sample[i], axis=0)

        else:
            sample = np.expand_dims(sample, axis=0)

        return sample

    def read_image(self, file_path: Path) -> np.ndarray:
        """
        Read image from source and correct dimensions.

        Parameters
        ----------
        file_path : Path
            Path to image source

        Returns
        -------
        image volume : np.ndarray
        """
        if not file_path.exists():
            raise ValueError(f"File {file_path} does not exist")

        try:
            sample = tifffile.imread(file_path)
            sample = sample.squeeze()
            sample = sample.astype(np.float32)
        except (ValueError, OSError) as e:
            logging.exception(f"Exception in file {file_path}: {e}, skipping")
            raise e

        # check number of dimensions
        if len(sample.shape) < 2 or len(sample.shape) > 4:
            raise ValueError(
                f"Incorrect data dimensions. Must be 2, 3 or 4 (got {sample.shape} for file {file_path})."
            )

        # check number of axes
        if len(self.axes) != len(sample.shape):
            raise ValueError(
                f"Incorrect axes length (got {self.axes} for file {file_path})."
            )

        return sample

    def generate_patches(self, sample):
        patches = None

        if self.patch_extraction_method == ExtractionStrategies.TILED:
            patches = extract_patches_predict(
                sample, patch_size=self.patch_size, overlaps=self.patch_overlap
            )

        elif self.patch_extraction_method == ExtractionStrategies.SEQUENTIAL:
            patches = extract_patches_sequential(sample, patch_size=self.patch_size)

        elif self.patch_extraction_method == ExtractionStrategies.RANDOM:
            patches = extract_patches_random(sample, patch_size=self.patch_size)

        if patches is None:
            raise ValueError("No patches generated")

        return patches

    def iterate_files(self) -> np.ndarray:
        """
        Iterate over data source and yield whole image. Optional transform is applied to the images.

        Yields
        ------
        np.ndarray
        """
        # When num_workers > 0, each worker process will have a different copy of the dataset object
        # Configuring each copy independently to avoid having duplicate data returned from the workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        for i, filename in enumerate(self.files):
            if i % num_workers == worker_id:
                sample = self.read_image(filename)
                sample = self.fix_axes(sample)
                yield sample

    def __iter__(self) -> np.ndarray:
        """
        Iterate over data source and yield single patch. Optional transform is applied to the patches.

        Yields
        ------
        np.ndarray
        """
        for sample in self.iterate_files():
            if self.patch_extraction_method:
                # TODO: move S and T unpacking logic from patch generator
                patches = self.generate_patches(sample)

                for patch in patches:
                    # TODO: remove this ugly workaround for normalizing 'prediction' patches
                    if isinstance(patch, tuple):
                        normalized_patch = normalize(patch[0], self.mean, self.std)
                        patch = (normalized_patch, *patch[1:])
                    else:
                        patch = normalize(patch, self.mean, self.std)

                    if self.patch_transform is not None:
                        patch = self.patch_transform(
                            patch, **self.patch_transform_params
                        )

                    yield patch

            else:
                # if S or T dims are not empty - assume every image is a separate sample in dim 0
                # TODO: is there always mean and std?
                for item in sample[0]:
                    item = np.expand_dims(item, (0, 1))
                    item = normalize(item, self.mean, self.std)
                    yield item


def get_tiff_dataset(stage, config):
    if stage in (ConfigStageEnum.TRAINING, ConfigStageEnum.EVALUATION):
        if config.training is None:
            raise ValueError("Training configuration is not defined.")

        if stage == ConfigStageEnum.TRAINING:
            data_path = config.data.training_path
        else:
            data_path = config.data.validation_path

        dataset = TiffDataset(
            data_path=data_path,  # TODO this can be None
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_extraction_method=config.training.extraction_strategy,
            patch_size=config.training.patch_size,
            patch_transform=default_manipulate,
            patch_transform_params={
                "mask_pixel_percentage": config.algorithm.masked_pixel_percentage
            },
        )
    elif stage == ConfigStageEnum.PREDICTION:
        if config.prediction is None:
            raise ValueError("Prediction configuration is not defined.")

        dataset = TiffDataset(
            data_path=config.data.prediction_path,
            axes=config.data.axes,
            mean=config.data.mean,
            std=config.data.std,
            patch_size=config.prediction.tile_shape,  # TODO this can be None
            patch_extraction_method=config.prediction.extraction_strategy,
            patch_transform=None,
        )
    else:
        raise ValueError(f"Stage {stage} not supported")

    return dataset
