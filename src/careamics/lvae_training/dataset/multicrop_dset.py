"""
Here, we have multiple folders, each containing images of a single channel.
"""

from collections import defaultdict
from functools import cache

import numpy as np

from .types import DataSplitType


def l2(x):
    return np.sqrt(np.mean(np.array(x) ** 2))


class MultiCropDset:
    def __init__(
        self,
        data_config,
        fpath: str,
        load_data_fn=None,
        val_fraction=None,
        test_fraction=None,
    ):

        assert (
            data_config.input_is_sum == True
        ), "This dataset is designed for sum of images"

        self._img_sz = data_config.image_size
        self._enable_rotation = data_config.enable_rotation_aug

        self._background_values = data_config.background_values
        self._data = load_data_fn(
            data_config, fpath, data_config.datasplit_type, val_fraction, test_fraction
        )

        # remove upper quantiles, crucial for removing puncta
        self.max_val = data_config.max_val
        if self.max_val is not None:
            for ch_idx, data in enumerate(self._data):
                if self.max_val[ch_idx] is not None:
                    for idx in range(len(data)):
                        data[idx][data[idx] > self.max_val[ch_idx]] = self.max_val[
                            ch_idx
                        ]

        # remove background values
        if self._background_values is not None:
            final_data_arr = []
            for ch_idx, data in enumerate(self._data):
                data_float = [x.astype(np.float32) for x in data]
                final_data_arr.append(
                    [x - self._background_values[ch_idx] for x in data_float]
                )
            self._data = final_data_arr

        print(
            f"{self.__class__.__name__} N:{len(self)} Rot:{self._enable_rotation} Ch:{len(self._data)} MaxVal:{self.max_val} Bg:{self._background_values}"
        )

    def get_max_val(self):
        return self.max_val

    def compute_mean_std(self):
        mean_tar_dict = defaultdict(list)
        std_tar_dict = defaultdict(list)
        mean_inp = []
        std_inp = []
        for _ in range(30000):
            crops = []
            for ch_idx in range(len(self._data)):
                crop = self.sample_crop(ch_idx)
                mean_tar_dict[ch_idx].append(np.mean(crop))
                std_tar_dict[ch_idx].append(np.std(crop))
                crops.append(crop)

            inp = 0
            for img in crops:
                inp += img

            mean_inp.append(np.mean(inp))
            std_inp.append(np.std(inp))

        output_mean = defaultdict(list)
        output_std = defaultdict(list)
        NC = len(self._data)
        for ch_idx in range(NC):
            output_mean["target"].append(np.mean(mean_tar_dict[ch_idx]))
            output_std["target"].append(l2(std_tar_dict[ch_idx]))

        output_mean["target"] = np.array(output_mean["target"]).reshape(NC, 1, 1)
        output_std["target"] = np.array(output_std["target"]).reshape(NC, 1, 1)

        output_mean["input"] = np.array([np.mean(mean_inp)]).reshape(1, 1, 1)
        output_std["input"] = np.array([l2(std_inp)]).reshape(1, 1, 1)
        return dict(output_mean), dict(output_std)

    def set_mean_std(self, mean_dict, std_dict):
        self._data_mean = mean_dict
        self._data_std = std_dict

    def get_mean_std(self):
        return self._data_mean, self._data_std

    def get_num_frames(self):
        return len(self._data)

    @cache
    def crop_probablities(self, ch_idx):
        sizes = np.array([np.prod(x.shape) for x in self._data[ch_idx]])
        return sizes / sizes.sum()

    def sample_crop(self, ch_idx):
        idx = None
        count = 0
        while idx is None:
            count += 1
            idx = np.random.choice(
                len(self._data[ch_idx]), p=self.crop_probablities(ch_idx)
            )
            data = self._data[ch_idx][idx]
            if data.shape[0] >= self._img_sz[0] and data.shape[1] >= self._img_sz[1]:
                h = np.random.randint(0, data.shape[0] - self._img_sz[0])
                w = np.random.randint(0, data.shape[1] - self._img_sz[1])
                return data[h : h + self._img_sz[0], w : w + self._img_sz[1]]
            elif count > 100:
                raise ValueError("Cannot find a valid crop")
            else:
                idx = None

        return None

    def len_per_channel(self, ch_idx):
        return np.sum([np.prod(x.shape) for x in self._data[ch_idx]]) / np.prod(
            self._img_sz
        )

    def imgs_for_patch(self):
        return [self.sample_crop(ch_idx) for ch_idx in range(len(self._data))]

    def __len__(self):
        len_per_channel = [
            self.len_per_channel(ch_idx) for ch_idx in range(len(self._data))
        ]
        return int(np.max(len_per_channel))

    def _rotate(self, img_tuples):
        return self._rotate2D(img_tuples)

    def _rotate2D(self, img_tuples):
        img_kwargs = {}
        for i, img in enumerate(img_tuples):
            for k in range(len(img)):
                img_kwargs[f"img{i}_{k}"] = img[k]

        keys = list(img_kwargs.keys())
        self._rotation_transform.add_targets({k: "image" for k in keys})
        rot_dic = self._rotation_transform(image=img_tuples[0][0], **img_kwargs)

        rotated_img_tuples = []
        for i, img in enumerate(img_tuples):
            if len(img) == 1:
                rotated_img_tuples.append(rot_dic[f"img{i}_0"][None])
            else:
                rotated_img_tuples.append(
                    np.concatenate(
                        [rot_dic[f"img{i}_{k}"][None] for k in range(len(img))], axis=0
                    )
                )

        return rotated_img_tuples

    def _compute_input(self, imgs):
        inp = 0
        for img in imgs:
            inp += img

        inp = inp[None]
        inp = (inp - self._data_mean["input"]) / (self._data_std["input"])
        return inp

    def _compute_target(self, imgs):
        imgs = np.stack(imgs)
        target = (imgs - self._data_mean["target"]) / (self._data_std["target"])
        return target

    def __getitem__(self, idx):
        imgs = self.imgs_for_patch()
        if self._enable_rotation:
            imgs = self._rotate(imgs)

        inp = self._compute_input(imgs)
        target = self._compute_target(imgs)
        return inp, target
