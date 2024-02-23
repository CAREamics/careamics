from __future__ import annotations
from pathlib import Path
import copy
from typing import Callable, Generator, List, Optional, Tuple, Union

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from ..config.data_model import DataModel
from ..config.support import SupportedExtractionStrategy
from ..utils.logging import get_logger
from .dataset_utils import read_tiff
from .patching import (
    get_patch_transform, 
    generate_patches_supervised, 
    generate_patches_unsupervised,
    generate_patches_predict
)

logger = get_logger(__name__)


class IterableDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    patch_extraction_method : Union[ExtractionStrategies, None]
        Patch extraction strategy, as defined in extraction_strategy.
    patch_size : Optional[Union[List[int], Tuple[int]]], optional
        Size of the patches in each dimension, by default None.
    patch_overlap : Optional[Union[List[int], Tuple[int]]], optional
        Overlap of the patches in each dimension, by default None.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        data_config: DataModel,
        src_files: List[Path],
        target_files: Optional[List[Path]] = None,
        read_source_func: Callable = read_tiff,
    ) -> None:

        if target_files is not None:
            raise NotImplementedError("Targets are not yet supported.")

        self.data_files = src_files
        self.target_files = target_files
        self.axes = data_config.axes
        self.patch_size = data_config.patch_size
        self.patch_extraction_method = SupportedExtractionStrategy.RANDOM
        self.read_source_func = read_source_func

        # compute mean and std over the dataset
        if not data_config.mean or not data_config.std:
            self.mean, self.std = self._calculate_mean_and_std()

            # if the transforms are not an instance of Compose
            if data_config.has_tranform_list():
                # update mean and std in configuration
                # the object is mutable and should then be recorded in the CAREamist obj
                data_config.set_mean_and_std(self.mean, self.std)

        # get transforms
        self.patch_transform = get_patch_transform(
            patch_transforms=data_config.transforms,
            with_target=target_files is not None,
        )

    def _calculate_mean_and_std(self) -> Tuple[float, float]:
        """
        Calculate mean and std of the dataset.

        Returns
        -------
        Tuple[float, float]
            Tuple containing mean and standard deviation.
        """
        means, stds = 0, 0
        num_samples = 0

        for sample in self._iterate_over_files():
            means += sample.mean()
            stds += sample.std()
            num_samples += 1

        if num_samples == 0:
            raise ValueError("No samples found in the dataset.")
        
        result_mean = means / num_samples
        result_std = stds / num_samples

        logger.info(f"Calculated mean and std for {num_samples} images")
        logger.info(f"Mean: {result_mean}, std: {result_std}")
        return result_mean, result_std

    def _iterate_over_files(self) -> Generator[Tuple[np.ndarray, ...], None, None]:
        """
        Iterate over data source and yield whole image.

        Yields
        ------
        np.ndarray
            Image.
        """
        # When num_workers > 0, each worker process will have a different copy of the
        # dataset object
        # Configuring each copy independently to avoid having duplicate data returned
        # from the workers
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        # iterate over the files
        for i, filename in enumerate(self.data_files):
            # retrieve file corresponding to the worker id
            if i % num_workers == worker_id:
                try: 
                    # read data
                    sample = self.read_source_func(filename, self.axes)

                    # TODO validation of the dimensions here

                    # read target if available
                    if self.target_files is not None:
                        if filename.name != self.target_files[i].name:
                            raise ValueError(
                                f"File {filename} does not match target file "
                                f"{self.target_files[i]}. Have you passed sorted "
                                f"arrays?"
                            )
                        
                        # read target
                        target = self.read_source_func(self.target_files[i], self.axes)
                        yield sample, target
                    else:
                        yield sample
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")                

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"

        # iterate over files
        for sample in self._iterate_over_files():
            if self.target_files is not None:
                patches = generate_patches_supervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                )

            else:
                patches = generate_patches_unsupervised(
                    sample,
                    self.axes,
                    self.patch_extraction_method,
                    self.patch_size,
                )

            # iterate over patches
            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    if self.target_files is not None:
                        target = patch_data[1:]
                        transformed = self.patch_transform(
                            image=np.moveaxis(patch_data[0], 0, -1),
                            target=np.moveaxis(target, 0, -1),
                        )
                        yield (transformed["image"], transformed["mask"])
                        # TODO fix dimensions
                    else:
                        # Albumentations expects the channel dimension to be last
                        patch = np.moveaxis(patch_data[0], 0, -1)

                        # apply transform
                        transformed = self.patch_transform(
                            image=patch
                        )
      
                        yield (transformed["image"], *patch_data[1:])
                else:
                    yield self.patch_transform(image=patch_data)["image"]

    def get_number_of_files(self) -> int:
        """
        Return the number of files in the dataset.

        Returns
        -------
        int
            Number of files in the dataset.
        """
        return len(self.data_files)

    def split_dataset(
            self, 
            percentage: float = 0.1,
            minimum_number: int = 5,
        ) -> IterableDataset: 
        
        if percentage < 0 or percentage > 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}.")
        
        if minimum_number < 1 or minimum_number > self.get_number_of_files():
            raise ValueError(
                f"Minimum number of files must be between 1 and "
                f"{self.get_number_of_files()} (number of files), got "
                f"{minimum_number}."
            )
        
        # compute number of files
        total_files = self.get_number_of_files()
        n_files = max(round(percentage*total_files), minimum_number)

        # get random indices
        indices = np.random.choice(total_files, n_files, replace=False)

        # extract files
        val_files = [self.data_files[i] for i in indices]

        # remove patches from self.patch
        data_files = []
        for i, file in enumerate(self.data_files):
            if i not in indices:
                data_files.append(file)
        self.data_files = data_files

        # same for targets
        if self.target_files is not None:
            val_target_files = [self.target_files[i] for i in indices]

            data_target_files = []
            for i, file in enumerate(self.target_files):
                if i not in indices:
                    data_target_files.append(file)
            self.target_files = data_target_files

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.data_files = val_files

        # reassign targets
        if self.target_files is not None:
            dataset.target_files = val_target_files

        return dataset


# TODO: why was this calling transforms on prediction patches?
class IterablePredictionDataset(IterableDataset):
    """
    Dataset allowing extracting patches w/o loading whole data into memory.

    Parameters
    ----------
    data_path : Union[str, Path]
        Path to the data, must be a directory.
    axes : str
        Description of axes in format STCZYX.
    patch_extraction_method : Union[ExtractionStrategies, None]
        Patch extraction strategy, as defined in extraction_strategy.
    patch_size : Optional[Union[List[int], Tuple[int]]], optional
        Size of the patches in each dimension, by default None.
    patch_overlap : Optional[Union[List[int], Tuple[int]]], optional
        Overlap of the patches in each dimension, by default None.
    mean : Optional[float], optional
        Expected mean of the dataset, by default None.
    std : Optional[float], optional
        Expected standard deviation of the dataset, by default None.
    patch_transform : Optional[Callable], optional
        Patch transform callable, by default None.
    """

    def __init__(
        self,
        data_config: DataModel,
        files: List[Path],
        tile_size: Union[List[int], Tuple[int]],
        tile_overlap: Optional[Union[List[int], Tuple[int]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs,
    ) -> None:
        super().__init__(
            data_config=data_config, 
            src_files=files, 
            read_source_func=read_source_func
        )
        
        self.patch_size = data_config.patch_size
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.read_source_func = read_source_func

        # check that mean and std are provided
        if not self.mean or not self.std:
            raise ValueError(
                f"Mean and std must be provided to the configuration in order to "
                f" perform prediction."
            )


    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """
        Iterate over data source and yield single patch.

        Yields
        ------
        np.ndarray
            Single patch.
        """
        assert (
            self.mean is not None and self.std is not None
        ), "Mean and std must be provided"

        for sample in self._iterate_over_files():
            patches =  generate_patches_predict(
                sample, self.axes, self.tile_size, self.tile_overlap
            )

            
            for patch_data in patches:
                if isinstance(patch_data, tuple):
                    transformed = self.patch_transform(
                        image=np.moveaxis(patch_data[0], 0, -1)
                    )
                    yield (np.moveaxis(transformed["image"], -1, 0), *patch_data[1:])
                else:
                    yield self.patch_transform(image=patch_data)["image"]
