"""
Utility functions needed by dataloader & co.
"""

import os
from typing import List

import numpy as np
from skimage.io import imread, imsave

from careamics.lvae_training.dataset.vae_data_config import DataSplitType, DataType


def load_tiff(path):
    """
    Returns a 4d numpy array: num_imgs*h*w*num_channels
    """
    data = imread(path, plugin="tifffile")
    return data


def save_tiff(path, data):
    imsave(path, data, plugin="tifffile")


def load_tiffs(paths):
    data = [load_tiff(path) for path in paths]
    return np.concatenate(data, axis=0)


def split_in_half(s, e):
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def adjust_for_imbalance_in_fraction_value(
    val: List[int],
    test: List[int],
    val_fraction: float,
    test_fraction: float,
    total_size: int,
):
    """
    here, val and test are divided almost equally. Here, we need to take into account their respective fractions
    and pick elements rendomly from one array and put in the other array.
    """
    if val_fraction == 0:
        test += val
        val = []
    elif test_fraction == 0:
        val += test
        test = []
    else:
        diff_fraction = test_fraction - val_fraction
        if diff_fraction > 0:
            imb_count = int(diff_fraction * total_size / 2)
            val = list(np.random.RandomState(seed=955).permutation(val))
            test += val[:imb_count]
            val = val[imb_count:]
        elif diff_fraction < 0:
            imb_count = int(-1 * diff_fraction * total_size / 2)
            test = list(np.random.RandomState(seed=955).permutation(test))
            val += test[:imb_count]
            test = test[imb_count:]
    return val, test


def get_train_val_data(
    data_config,
    fpath,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    allow_generation=False,  # TODO: what is this
):
    """
    Load the data from the given path and split them in training, validation and test sets.

    Ensure that the shape of data should be N*H*W*C: N is number of data points. H,W are the image dimensions.
    C is the number of channels.
    """
    if data_config.data_type == DataType.SeparateTiffData:
        fpath1 = os.path.join(fpath, data_config.ch1_fname)
        fpath2 = os.path.join(fpath, data_config.ch2_fname)
        fpaths = [fpath1, fpath2]
        fpath0 = ""
        if "ch_input_fname" in data_config:
            fpath0 = os.path.join(fpath, data_config.ch_input_fname)
            fpaths = [fpath0] + fpaths

        print(
            f"Loading from {fpath} Channels: "
            f"{fpath1},{fpath2}, inp:{fpath0} Mode:{DataSplitType.name(datasplit_type)}"
        )

        data = np.concatenate([load_tiff(fpath)[..., None] for fpath in fpaths], axis=3)
        if data_config.data_type == DataType.PredictedTiffData:
            assert len(data.shape) == 5 and data.shape[-1] == 1
            data = data[..., 0].copy()
        # data = data[::3].copy()
        # NOTE: This was not the correct way to do it. It is so because the noise present in the input was directly related
        # to the noise present in the channels and so this is not the way we would get the data.
        # We need to add the noise independently to the input and the target.

        # if data_config.get('poisson_noise_factor', False):
        #     data = np.random.poisson(data)
        # if data_config.get('enable_gaussian_noise', False):
        #     synthetic_scale = data_config.get('synthetic_gaussian_scale', 0.1)
        #     print('Adding Gaussian noise with scale', synthetic_scale)
        #     noise = np.random.normal(0, synthetic_scale, data.shape)
        #     data = data + noise

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)

    elif data_config.data_type == DataType.BioSR_MRC:
        num_channels = data_config.num_channels
        fpaths = []
        data_list = []
        for i in range(num_channels):
            fpath1 = os.path.join(fpath, getattr(data_config, f"ch{i + 1}_fname"))
            fpaths.append(fpath1)
            data = get_mrc_data(fpath1)[..., None]
            data_list.append(data)

        dirname = os.path.dirname(os.path.dirname(fpaths[0])) + "/"

        msg = ",".join([x[len(dirname) :] for x in fpaths])
        print(
            f"Loaded from {dirname} Channels:{len(fpaths)} {msg} Mode:{datasplit_type}"
        )
        N = data_list[0].shape[0]
        for data in data_list:
            N = min(N, data.shape[0])

        cropped_data = []
        for data in data_list:
            cropped_data.append(data[:N])

        data = np.concatenate(cropped_data, axis=3)

        if datasplit_type == DataSplitType.All:
            return data.astype(np.float32)

        train_idx, val_idx, test_idx = get_datasplit_tuples(
            val_fraction, test_fraction, len(data), starting_test=True
        )
        if datasplit_type == DataSplitType.Train:
            return data[train_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Val:
            return data[val_idx].astype(np.float32)
        elif datasplit_type == DataSplitType.Test:
            return data[test_idx].astype(np.float32)


def get_datasplit_tuples(
    val_fraction: float,
    test_fraction: float,
    total_size: int,
    starting_test: bool = False,
):
    if starting_test:
        # test => val => train
        test = list(range(0, int(total_size * test_fraction)))
        val = list(range(test[-1] + 1, test[-1] + 1 + int(total_size * val_fraction)))
        train = list(range(val[-1] + 1, total_size))
    else:
        # {test,val}=> train
        test_val_size = int((val_fraction + test_fraction) * total_size)
        train = list(range(test_val_size, total_size))

        if test_val_size == 0:
            test = []
            val = []
            return train, val, test

        # Split the test and validation in chunks.
        chunksize = max(1, min(3, test_val_size // 2))

        nchunks = test_val_size // chunksize

        test = []
        val = []
        s = 0
        for i in range(nchunks):
            if i % 2 == 0:
                val += list(np.arange(s, s + chunksize))
            else:
                test += list(np.arange(s, s + chunksize))
            s += chunksize

        if i % 2 == 0:
            test += list(np.arange(s, test_val_size))
        else:
            p1, p2 = split_in_half(s, test_val_size)
            test += p1
            val += p2

    val, test = adjust_for_imbalance_in_fraction_value(
        val, test, val_fraction, test_fraction, total_size
    )

    return train, val, test


def get_mrc_data(fpath):
    # HXWXN
    _, data = read_mrc(fpath)
    data = data[None]
    data = np.swapaxes(data, 0, 3)
    return data[..., 0]


rec_header_dtd = [
    ("nx", "i4"),  # Number of columns
    ("ny", "i4"),  # Number of rows
    ("nz", "i4"),  # Number of sections
    ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
    #  0 = unsigned or signed bytes depending on flag in imodFlags
    #  1 = signed short integers (16 bits)
    #  2 = float (32 bits)
    #  3 = short * 2, (used for complex data)
    #  4 = float * 2, (used for complex data)
    #  6 = unsigned 16-bit integers (non-standard)
    # 16 = unsigned char * 3 (for rgb data, non-standard)
    ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
    ("nystart", "i4"),
    ("nzstart", "i4"),
    ("mx", "i4"),  # Grid size in X, Y and Z
    ("my", "i4"),
    ("mz", "i4"),
    ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
    ("ylen", "f4"),
    ("zlen", "f4"),
    ("alpha", "f4"),  # Cell angles - ignored by IMOD
    ("beta", "f4"),
    ("gamma", "f4"),
    # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
    ("mapc", "i4"),  # map column  1=x,2=y,3=z.
    ("mapr", "i4"),  # map row     1=x,2=y,3=z.
    ("maps", "i4"),  # map section 1=x,2=y,3=z.
    # These need to be set for proper scaling of data
    ("amin", "f4"),  # Minimum pixel value
    ("amax", "f4"),  # Maximum pixel value
    ("amean", "f4"),  # Mean pixel value
    ("ispg", "i4"),  # space group number (ignored by IMOD)
    (
        "next",
        "i4",
    ),  # number of bytes in extended header (called nsymbt in MRC standard)
    ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
    ("extra_data", "V30"),  # (not used, first two bytes should be 0)
    # These two values specify the structure of data in the extended header; their meaning depend on whether the
    # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
    # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
    # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
    ("nint", "i2"),
    # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
    ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
    # Number of reals per section (Agard format) or bit
    # flags for which types of short data (SerialEM format):
    # 1 = tilt angle * 100  (2 bytes)
    # 2 = piece coordinates for montage  (6 bytes)
    # 4 = Stage position * 25    (4 bytes)
    # 8 = Magnification / 100 (2 bytes)
    # 16 = Intensity * 25000  (2 bytes)
    # 32 = Exposure dose in e-/A2, a float in 4 bytes
    # 128, 512: Reserved for 4-byte items
    # 64, 256, 1024: Reserved for 2-byte items
    # If the number of bytes implied by these flags does
    # not add up to the value in nint, then nint and nreal
    # are interpreted as ints and reals per section
    ("extra_data2", "V20"),  # extra data (not used)
    ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
    ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed
    # Explanation of type of data
    ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
    ("lens", "i2"),
    # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
    # ("nd2", "i2"),
    ("nphase", "i4"),
    ("vd1", "i2"),  # vd1 = 100. * tilt increment
    ("vd2", "i2"),  # vd2 = 100. * starting angle
    # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
    # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
    ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current
    ("xorg", "f4"),  # Origin of image
    ("yorg", "f4"),
    ("zorg", "f4"),
    ("cmap", "S4"),  # Contains "MAP "
    (
        "stamp",
        "u1",
        4,
    ),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian
    ("rms", "f4"),  # RMS deviation of densities from mean density
    ("nlabl", "i4"),  # Number of labels with useful data
    ("labels", "S80", 10),  # 10 labels of 80 charactors
]


def read_mrc(filename, filetype="image"):
    fd = open(filename, "rb")
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header["nx"][0], header["ny"][0], header["nz"][0]

    if header[0][3] == 1:
        data_type = "int16"
    elif header[0][3] == 2:
        data_type = "float32"
    elif header[0][3] == 4:
        data_type = "single"
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = "uint16"

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == "image":
        for iz in range(nz):
            data_2d = imgrawdata[nx * ny * iz : nx * ny * (iz + 1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order="F")
    else:
        data = imgrawdata

    return header, data
