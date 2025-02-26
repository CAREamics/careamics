from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
from torch.utils.data import Dataset

from careamics.config import DataConfig
from careamics.config.transformations import NormalizeModel
from careamics.dataset.patching.patching import (
    PatchedOutput,
    Stats,
    prepare_patches_supervised,
    prepare_patches_supervised_array,
    prepare_patches_unsupervised,
    prepare_patches_unsupervised_array,
)
from careamics.file_io.read import read_tiff
from careamics.transforms import Compose
from careamics.utils.logging import get_logger

logger = get_logger(__name__)


from collections import defaultdict
from functools import cache

import numpy as np


def l2(x):
    return np.sqrt(np.mean(np.array(x)**2))


class MultiCropDset:
    def __init__(self,
                 data_config,
                 fpath: str,
                 datasplit_type: DataSplitType = None,
                 val_fraction=None,
                 test_fraction=None,
                 enable_rotation_aug: bool = False,
                 **ignore_kwargs):
        
        self._img_sz = data_config.image_size
        self._background_values = data_config.get('background_values', None)
        self._data_arr = get_train_val_data(data_config,fpath, datasplit_type, val_fraction, test_fraction)

        # remove upper quantiles, crucial for removing puncta
        self.max_val = data_config.get('max_val', None)
        if self.max_val is not None:
            for ch_idx, data in enumerate(self._data_arr):
                if self.max_val[ch_idx] is not None:
                    for idx in range(len(data)):
                        data[idx][data[idx] > self.max_val[ch_idx]] = self.max_val[ch_idx]

        # remove background values
        if self._background_values is not None:
            final_data_arr = []
            for ch_idx, data in enumerate(self._data_arr):
                data_float = [x.astype(np.float32) for x in data]
                final_data_arr.append([x - self._background_values[ch_idx] for x in data_float])
            self._data_arr = final_data_arr

        self.data_config = data_config
        self.inputs = inputs
        self.input_targets = input_target
        self.axes = self.data_config.axes
        self.patch_size = self.data_config.patch_size

        # read function
        self.read_source_func = read_source_func

        # generate patches
        supervised = self.input_targets is not None
        patches_data = self._prepare_patches(supervised)

        # unpack the dataclass
        self.data = patches_data.patches
        self.data_targets = patches_data.targets

        # set image statistics
        if self.data_config.image_means is None:
            self.image_stats = patches_data.image_stats
            logger.info(
                f"Computed dataset mean: {self.image_stats.means}, "
                f"std: {self.image_stats.stds}"
            )
        else:
            self.image_stats = Stats(
                self.data_config.image_means, self.data_config.image_stds
            )

        # set target statistics
        if self.data_config.target_means is None:
            self.target_stats = patches_data.target_stats
        else:
            self.target_stats = Stats(
                self.data_config.target_means, self.data_config.target_stds
            )

        # update mean and std in configuration
        # the object is mutable and should then be recorded in the CAREamist obj
        self.data_config.set_means_and_stds(
            image_means=self.image_stats.means,
            image_stds=self.image_stats.stds,
            target_means=self.target_stats.means,
            target_stds=self.target_stats.stds,
        )
        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(
                    image_means=self.image_stats.means,
                    image_stds=self.image_stats.stds,
                    target_means=self.target_stats.means,
                    target_stds=self.target_stats.stds,
                )
            ]
            + list(self.data_config.transforms),
        )

    @cache
    def crop_probablities(self, ch_idx):
        sizes = np.array([np.prod(x.shape) for x in self._data_arr[ch_idx]])
        return sizes/sizes.sum()
    
    def sample_crop(self, ch_idx):
        idx = None
        count = 0
        while idx is None:
            count += 1
            idx = np.random.choice(len(self._data_arr[ch_idx]), p=self.crop_probablities(ch_idx))
            data = self._data_arr[ch_idx][idx]
            if data.shape[0] >= self._img_sz and data.shape[1] >= self._img_sz:
                h = np.random.randint(0, data.shape[0] - self._img_sz)
                w = np.random.randint(0, data.shape[1] - self._img_sz)
                return data[h:h+self._img_sz, w:w+self._img_sz]
            elif count > 100:
                raise ValueError("Cannot find a valid crop")
            else:
                idx = None
        
        return None

    
    def len_per_channel(self, ch_idx):
        return np.sum([np.prod(x.shape) for x in self._data_arr[ch_idx]])/(self._img_sz*self._img_sz)
    
    def imgs_for_patch(self):
        return [self.sample_crop(ch_idx) for ch_idx in range(len(self._data_arr))]

    def __len__(self):
        len_per_channel = [self.len_per_channel(ch_idx) for ch_idx in range(len(self._data_arr))]
        return int(np.max(len_per_channel))

    def _rotate(self, img_tuples):
        return self._rotate2D(img_tuples)

    def _rotate2D(self, img_tuples):
        img_kwargs = {}
        for i,img in enumerate(img_tuples):
            img_kwargs[f'img{i}'] = img
        
        
        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: 'image' for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0], **img_kwargs)

        rotated_img_tuples = []
        for i,img in enumerate(img_tuples):
            rotated_img_tuples.append(rot_dic[f'img{i}'])

        
        return rotated_img_tuples

    def _compute_input(self, imgs):
        inp = 0
        for img in imgs:
            inp += img
        
        inp = (inp - self._data_mean['input'].squeeze())/(self._data_std['input'].squeeze())
        return inp[None]

    def _compute_target(self, imgs):
        return np.stack(imgs)

    def __getitem__(self, idx):
        imgs = self.imgs_for_patch()
        if self._enable_rotation:
            imgs = self._rotate(imgs)
        

        inp = self._compute_input(imgs)
        target = self._compute_target(imgs)
        return inp, target


class InMemoryDataset(Dataset):
    """Dataset storing data in memory and allowing generating patches from it.

    Parameters
    ----------
    data_config : CAREamics DataConfig
        (see careamics.config.data_model.DataConfig)
        Data configuration.
    inputs : numpy.ndarray or list[pathlib.Path]
        Input data.
    input_target : numpy.ndarray or list[pathlib.Path], optional
        Target data, by default None.
    read_source_func : Callable, optional
        Read source function for custom types, by default read_tiff.
    **kwargs : Any
        Additional keyword arguments, unused.
    """

    def __init__(
        self,
        data_config: DataConfig,
        inputs: Union[np.ndarray, list[Path]],
        input_target: Optional[Union[np.ndarray, list[Path]]] = None,
        read_source_func: Callable = read_tiff,
        **kwargs: Any,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_config : GeneralDataConfig
            Data configuration.
        inputs : numpy.ndarray or list[pathlib.Path]
            Input data.
        input_target : numpy.ndarray or list[pathlib.Path], optional
            Target data, by default None.
        read_source_func : Callable, optional
            Read source function for custom types, by default read_tiff.
        **kwargs : Any
            Additional keyword arguments, unused.
        """
        self.data_config = data_config
        self.inputs = inputs
        self.input_targets = input_target
        self.axes = self.data_config.axes
        self.patch_size = self.data_config.patch_size

        # read function
        self.read_source_func = read_source_func

        # generate patches
        supervised = self.input_targets is not None
        patches_data = self._prepare_patches(supervised)

        # unpack the dataclass
        self.data = patches_data.patches
        self.data_targets = patches_data.targets

        # set image statistics
        if self.data_config.image_means is None:
            self.image_stats = patches_data.image_stats
            logger.info(
                f"Computed dataset mean: {self.image_stats.means}, "
                f"std: {self.image_stats.stds}"
            )
        else:
            self.image_stats = Stats(
                self.data_config.image_means, self.data_config.image_stds
            )

        # set target statistics
        if self.data_config.target_means is None:
            self.target_stats = patches_data.target_stats
        else:
            self.target_stats = Stats(
                self.data_config.target_means, self.data_config.target_stds
            )

        # update mean and std in configuration
        # the object is mutable and should then be recorded in the CAREamist obj
        self.data_config.set_means_and_stds(
            image_means=self.image_stats.means,
            image_stds=self.image_stats.stds,
            target_means=self.target_stats.means,
            target_stds=self.target_stats.stds,
        )
        # get transforms
        self.patch_transform = Compose(
            transform_list=[
                NormalizeModel(
                    image_means=self.image_stats.means,
                    image_stds=self.image_stats.stds,
                    target_means=self.target_stats.means,
                    target_stds=self.target_stats.stds,
                )
            ]
            + list(self.data_config.transforms),
        )

    def _prepare_patches(self, supervised: bool) -> PatchedOutput:
        """
        Iterate over data source and create an array of patches.

        Parameters
        ----------
        supervised : bool
            Whether the dataset is supervised or not.

        Returns
        -------
        numpy.ndarray
            Array of patches.
        """
        if supervised:
            return prepare_patches_supervised(
                    self.inputs,
                    self.input_targets,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
        else:
            return prepare_patches_unsupervised(
                    self.inputs,
                    self.axes,
                    self.patch_size,
                    self.read_source_func,
                )
        # TODO patches should contain LC inputs, so have extra dimension for that.
        # extract patches should be modified to handle this.

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        """
        Return the patch corresponding to the provided index.

        Parameters
        ----------
        index : int
            Index of the patch to return.

        Returns
        -------
        tuple of numpy.ndarray
            Patch.

        Raises
        ------
        ValueError
            If dataset mean and std are not set.
        """
        patch = self.data[index]

        # if there is a target
        if self.data_targets is not None:
            # get target
            target = self.data_targets[index]
            return self.patch_transform(patch=patch, target=target)

        return self.patch_transform(patch=patch)

    def get_data_statistics(self) -> tuple[list[float], list[float]]:
        """Return training data statistics.

        This does not return the target data statistics, only those of the input.

        Returns
        -------
        tuple of list of floats
            Means and standard deviations across channels of the training data.
        """
        return self.image_stats.get_statistics()

    def split_dataset(
        self,
        percentage: float = 0.1,
        minimum_patches: int = 1,
    ) -> InMemoryDataset:
        """Split a new dataset away from the current one.

        This method is used to extract random validation patches from the dataset.

        Parameters
        ----------
        percentage : float, optional
            Percentage of patches to extract, by default 0.1.
        minimum_patches : int, optional
            Minimum number of patches to extract, by default 5.

        Returns
        -------
        CAREamics InMemoryDataset
            New dataset with the extracted patches.

        Raises
        ------
        ValueError
            If `percentage` is not between 0 and 1.
        ValueError
            If `minimum_number` is not between 1 and the number of patches.
        """
        if percentage < 0 or percentage > 1:
            raise ValueError(f"Percentage must be between 0 and 1, got {percentage}.")

        if minimum_patches < 1 or minimum_patches > len(self):
            raise ValueError(
                f"Minimum number of patches must be between 1 and "
                f"{len(self)} (number of patches), got "
                f"{minimum_patches}. Adjust the patch size or the minimum number of "
                f"patches."
            )

        total_patches = len(self)

        # number of patches to extract (either percentage rounded or minimum number)
        n_patches = max(round(total_patches * percentage), minimum_patches)

        # get random indices
        indices = np.random.choice(total_patches, n_patches, replace=False)

        # extract patches
        val_patches = self.data[indices]

        # remove patches from self.patch
        self.data = np.delete(self.data, indices, axis=0)

        # same for targets
        if self.data_targets is not None:
            val_targets = self.data_targets[indices]
            self.data_targets = np.delete(self.data_targets, indices, axis=0)

        # clone the dataset
        dataset = copy.deepcopy(self)

        # reassign patches
        dataset.data = val_patches

        # reassign targets
        if self.data_targets is not None:
            dataset.data_targets = val_targets

        return dataset


class LCMultiChDloader(MultiChDloader):
    def __init__(
        self,
        data_config: DatasetConfig,
        fpath: str,
        load_data_fn: Callable,
        val_fraction=None,
        test_fraction=None,
    ):
        self._padding_kwargs = (
            data_config.padding_kwargs  # mode=padding_mode, constant_values=constant_value
        )
        self._uncorrelated_channel_probab = data_config.uncorrelated_channel_probab

        super().__init__(
            data_config,
            fpath,
            load_data_fn=load_data_fn,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        if data_config.overlapping_padding_kwargs is not None:
            assert (
                self._padding_kwargs == data_config.overlapping_padding_kwargs
            ), "During evaluation, overlapping_padding_kwargs should be same as padding_args. \
                It should be so since we just use overlapping_padding_kwargs when it is not None"

        else:
            self._overlapping_padding_kwargs = data_config.padding_kwargs

        self.multiscale_lowres_count = data_config.multiscale_lowres_count
        assert self.multiscale_lowres_count is not None
        self._scaled_data = [self._data]
        self._scaled_noise_data = [self._noise_data]

        assert (
            isinstance(self.multiscale_lowres_count, int)
            and self.multiscale_lowres_count >= 1
        )
        assert isinstance(self._padding_kwargs, dict)
        assert "mode" in self._padding_kwargs

        for _ in range(1, self.multiscale_lowres_count):
            shape = self._scaled_data[-1].shape
            assert len(shape) == 4
            new_shape = (shape[0], shape[1] // 2, shape[2] // 2, shape[3])
            ds_data = resize(
                self._scaled_data[-1].astype(np.float32), new_shape
            ).astype(self._scaled_data[-1].dtype)
            # NOTE: These asserts are important. the resize method expects np.float32. otherwise, one gets weird results.
            assert (
                ds_data.max() / self._scaled_data[-1].max() < 5
            ), "Downsampled image should not have very different values"
            assert (
                ds_data.max() / self._scaled_data[-1].max() > 0.2
            ), "Downsampled image should not have very different values"

            self._scaled_data.append(ds_data)
            # do the same for noise
            if self._noise_data is not None:
                noise_data = resize(self._scaled_noise_data[-1], new_shape)
                self._scaled_noise_data.append(noise_data)

    def reduce_data(
        self, t_list=None, h_start=None, h_end=None, w_start=None, w_end=None
    ):
        assert t_list is not None
        assert h_start is None
        assert h_end is None
        assert w_start is None
        assert w_end is None

        self._data = self._data[t_list].copy()
        self._scaled_data = [
            self._scaled_data[i][t_list].copy() for i in range(len(self._scaled_data))
        ]

        if self._noise_data is not None:
            self._noise_data = self._noise_data[t_list].copy()
            self._scaled_noise_data = [
                self._scaled_noise_data[i][t_list].copy()
                for i in range(len(self._scaled_noise_data))
            ]

        self.N = len(t_list)
        # TODO where tf is self._img_sz defined?
        self.set_img_sz([self._img_sz, self._img_sz], self._grid_sz)
        print(
            f"[{self.__class__.__name__}] Data reduced. New data shape: {self._data.shape}"
        )

    def _init_msg(self):
        msg = super()._init_msg()
        msg += f" Pad:{self._padding_kwargs}"
        if self._uncorrelated_channels:
            msg += f" UncorrChProbab:{self._uncorrelated_channel_probab}"
        return msg

    def _load_scaled_img(
        self, scaled_index, index: Union[int, Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(index, int):
            idx = index
        else:
            idx, _ = index

        # tidx = self.idx_manager.get_t(idx)
        patch_loc_list = self.idx_manager.get_patch_location_from_dataset_idx(idx)
        nidx = patch_loc_list[0]

        imgs = self._scaled_data[scaled_index][nidx]
        imgs = tuple([imgs[None, ..., i] for i in range(imgs.shape[-1])])
        if self._noise_data is not None:
            noisedata = self._scaled_noise_data[scaled_index][nidx]
            noise = tuple([noisedata[None, ..., i] for i in range(noisedata.shape[-1])])
            factor = np.sqrt(2) if self._input_is_sum else 1.0
            imgs = tuple([img + noise[0] * factor for img in imgs])
        return imgs

    def _crop_img(self, img: np.ndarray, patch_start_loc: Tuple):
        """
        Here, h_start, w_start could be negative. That simply means we need to pick the content from 0. So,
        the cropped image will be smaller than self._img_sz * self._img_sz
        """
        max_len_vals = list(self.idx_manager.data_shape[1:-1])
        max_len_vals[-2:] = img.shape[-2:]
        return self._crop_img_with_padding(
            img, patch_start_loc, max_len_vals=max_len_vals
        )

    def _get_img(self, index: int):
        """
        Returns the primary patch along with low resolution patches centered on the primary patch.
        """
        # Noise_tuples is populated when there is synthetic noise in training
        # Should have similar type of noise with the noise model
        # Starting with microsplit, dump the noise, use it instead as an augmentation if nessesary
        img_tuples, noise_tuples = self._load_img(index)
        assert self._img_sz is not None
        h, w = img_tuples[0].shape[-2:]
        if self._enable_random_cropping:
            patch_start_loc = self._get_random_hw(h, w)
            if self._5Ddata:
                patch_start_loc = (
                    np.random.choice(img_tuples[0].shape[-3] - self._depth3D),
                ) + patch_start_loc
        else:
            patch_start_loc = self._get_deterministic_loc(index)

        # LC logic is located here, the function crops the image of the highest resolution
        cropped_img_tuples = [
            self._crop_flip_img(img, patch_start_loc, False, False)
            for img in img_tuples
        ]
        cropped_noise_tuples = [
            self._crop_flip_img(noise, patch_start_loc, False, False)
            for noise in noise_tuples
        ]
        patch_start_loc = list(patch_start_loc)
        h_start, w_start = patch_start_loc[-2], patch_start_loc[-1]
        h_center = h_start + self._img_sz // 2
        w_center = w_start + self._img_sz // 2
        allres_versions = {
            i: [cropped_img_tuples[i]] for i in range(len(cropped_img_tuples))
        }
        for scale_idx in range(1, self.multiscale_lowres_count):
            # Returning the image of the lower resolution
            scaled_img_tuples = self._load_scaled_img(scale_idx, index)

            h_center = h_center // 2
            w_center = w_center // 2

            h_start = h_center - self._img_sz // 2
            w_start = w_center - self._img_sz // 2
            patch_start_loc[-2:] = [h_start, w_start]
            scaled_cropped_img_tuples = [
                self._crop_flip_img(img, patch_start_loc, False, False)
                for img in scaled_img_tuples
            ]
            for ch_idx in range(len(img_tuples)):
                allres_versions[ch_idx].append(scaled_cropped_img_tuples[ch_idx])

        output_img_tuples = tuple(
            [
                np.concatenate(allres_versions[ch_idx])
                for ch_idx in range(len(img_tuples))
            ]
        )
        return output_img_tuples, cropped_noise_tuples

    
    def _compute_target(self, img_tuples, alpha):
        if self._tar_idx_list is not None and isinstance(self._tar_idx_list, int):
            target = img_tuples[self._tar_idx_list]
        else:
            if self._tar_idx_list is not None:
                assert isinstance(self._tar_idx_list, list) or isinstance(
                    self._tar_idx_list, tuple
                )
                img_tuples = [img_tuples[i] for i in self._tar_idx_list]

            target = np.concatenate(img_tuples, axis=0)
        return target

    def _compute_input_with_alpha(self, img_tuples, alpha_list):
        # assert self._normalized_input is True, "normalization should happen here"
        if self._input_idx is not None:
            inp = img_tuples[self._input_idx]
        else:
            inp = 0
            for alpha, img in zip(alpha_list, img_tuples):
                inp += img * alpha

            if self._normalized_input is False:
                return inp.astype(np.float32)

        mean, std = self.get_mean_std_for_input()
        mean = mean.squeeze()
        std = std.squeeze()
        if mean.size == 1:
            mean = mean.reshape(
                1,
            )
            std = std.reshape(
                1,
            )

        for i in range(len(mean)):
            assert mean[0] == mean[i]
            assert std[0] == std[i]

        inp = (inp - mean[0]) / std[0]
        return inp.astype(np.float32)

    def _sample_alpha(self):
        alpha_arr = []
        for i in range(self._num_channels):
            alpha_pos = np.random.rand()
            alpha = self._start_alpha_arr[i] + alpha_pos * (
                self._end_alpha_arr[i] - self._start_alpha_arr[i]
            )
            alpha_arr.append(alpha)
        return alpha_arr

    def _compute_input(self, img_tuples):
        alpha = [1 / len(img_tuples) for _ in range(len(img_tuples))]
        if self._start_alpha_arr is not None:
            alpha = self._sample_alpha()

        inp = self._compute_input_with_alpha(img_tuples, alpha)
        if self._input_is_sum:
            inp = len(img_tuples) * inp
        return inp, alpha

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        img_tuples, noise_tuples = self._get_img(index)

        if self._is_train:
            if self._empty_patch_replacement_enabled:
                if np.random.rand() < self._empty_patch_replacement_probab:
                    img_tuples = self.replace_with_empty_patch(img_tuples)

        if self._enable_rotation:
            input_tuples, noise_tuples = self._rotate(img_tuples, noise_tuples)

        # Compute the input by sum / average the channels
        # Alpha is an amount of weight which is applied to the channels when combining them
        # How to sample alpha is still under research
        inp, alpha = self._compute_input(input_tuples)
        target_tuples = [img[:1] for img in img_tuples]
        # add noise to target.
        if len(noise_tuples) >= 1:
            target_tuples = [
                x + noise for x, noise in zip(target_tuples, noise_tuples[1:])
            ]

        target = self._compute_target(target_tuples, alpha)

        norm_target = self.normalize_target(target)

        output = [inp, norm_target]

        if self._return_alpha:
            output.append(alpha)

        if isinstance(index, int):
            return tuple(output)

        _, grid_size = index
        output.append(grid_size)
        return tuple(output)
