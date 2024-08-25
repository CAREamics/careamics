"""
Utility functions needed by dataloader & co.
"""

from typing import List

import numpy as np
from skimage.io import imread, imsave
from careamics.models.lvae.utils import Enum


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


class GridIndexManager:

    def __init__(self, data_shape, grid_size, patch_size, grid_alignement) -> None:
        self._data_shape = data_shape
        self._default_grid_size = grid_size
        self.patch_size = patch_size
        self.N = self._data_shape[0]
        self._align = grid_alignement

    def get_data_shape(self):
        return self._data_shape

    def use_default_grid(self, grid_size):
        return grid_size is None or grid_size < 0

    def grid_rows(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = self.patch_size - grid_size
        elif self._align == GridAlignement.Center:
            # Center is exclusively used during evaluation. In this case, we use the padding to handle edge cases.
            # So, here, we will ideally like to cover all pixels and so extra_pixels is set to 0.
            # If there was no padding, then it should be set to (self.patch_size - grid_size) // 2
            extra_pixels = 0

        return (self._data_shape[-3] - extra_pixels) // grid_size

    def grid_cols(self, grid_size):
        if self._align == GridAlignement.LeftTop:
            extra_pixels = self.patch_size - grid_size
        elif self._align == GridAlignement.Center:
            extra_pixels = 0

        return (self._data_shape[-2] - extra_pixels) // grid_size

    def grid_count(self, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        return self.N * self.grid_rows(grid_size) * self.grid_cols(grid_size)

    def hwt_from_idx(self, index, grid_size=None):
        t = self.get_t(index)
        return (*self.get_deterministic_hw(index, grid_size=grid_size), t)

    def idx_from_hwt(self, h_start, w_start, t, grid_size=None):
        """
        Given h,w,t (where h,w constitutes the top left corner of the patch), it returns the corresponding index.
        """
        if grid_size is None:
            grid_size = self._default_grid_size

        nth_row = h_start // grid_size
        nth_col = w_start // grid_size

        index = self.grid_cols(grid_size) * nth_row + nth_col
        return index * self._data_shape[0] + t

    def get_t(self, index):
        return index % self.N

    def get_top_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index -= ncols * self.N
        if index < 0:
            return None

        return index

    def get_bottom_nbr_idx(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        index += ncols * self.N
        if index > self.grid_count(grid_size=grid_size):
            return None

        return index

    def get_left_nbr_idx(self, index, grid_size=None):
        if self.on_left_boundary(index, grid_size=grid_size):
            return None

        index -= self.N
        return index

    def get_right_nbr_idx(self, index, grid_size=None):
        if self.on_right_boundary(index, grid_size=grid_size):
            return None
        index += self.N
        return index

    def on_left_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        left_boundary = (factor // ncols) != (factor - 1) // ncols
        return left_boundary

    def on_right_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        right_boundary = (factor // ncols) != (factor + 1) // ncols
        return right_boundary

    def on_top_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index < self.N * ncols

    def on_bottom_boundary(self, index, grid_size=None):
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        ncols = self.grid_cols(grid_size)
        return index + self.N * ncols > self.grid_count(grid_size=grid_size)

    def on_boundary(self, idx, grid_size=None):
        if self.on_left_boundary(idx, grid_size=grid_size):
            return True

        if self.on_right_boundary(idx, grid_size=grid_size):
            return True

        if self.on_top_boundary(idx, grid_size=grid_size):
            return True

        if self.on_bottom_boundary(idx, grid_size=grid_size):
            return True
        return False

    def get_deterministic_hw(self, index: int, grid_size=None):
        """
        Fixed starting position for the crop for the img with index `index`.
        """
        if self.use_default_grid(grid_size):
            grid_size = self._default_grid_size

        # _, h, w, _ = self._data_shape
        # assert h == w
        factor = index // self.N
        ncols = self.grid_cols(grid_size)

        ith_row = factor // ncols
        jth_col = factor % ncols
        h_start = ith_row * grid_size
        w_start = jth_col * grid_size
        return h_start, w_start


class IndexSwitcher:
    """
    The idea is to switch from valid indices for target to invalid indices for target.
    If index in invalid for the target, then we return all zero vector as target.
    This combines both logic:
    1. Using less amount of total data.
    2. Using less amount of target data but using full data.
    """

    def __init__(self, idx_manager, data_config, patch_size) -> None:
        self.idx_manager = idx_manager
        self._data_shape = self.idx_manager.get_data_shape()
        self._training_validtarget_fraction = data_config.get(
            "training_validtarget_fraction", 1.0
        )
        self._validtarget_ceilT = int(
            np.ceil(self._data_shape[0] * self._training_validtarget_fraction)
        )
        self._patch_size = patch_size
        assert (
            data_config.deterministic_grid is True
        ), "This only works when the dataset has deterministic grid. Needed randomness comes from this class."
        assert (
            "grid_size" in data_config and data_config.grid_size == 1
        ), "We need a one to one mapping between index and h, w, t"

        self._h_validmax, self._w_validmax = self.get_reduced_frame_size(
            self._data_shape[:3], self._training_validtarget_fraction
        )
        if self._h_validmax < self._patch_size or self._w_validmax < self._patch_size:
            print(
                "WARNING: The valid target size is smaller than the patch size. This will result in all zero target. so, we are ignoring this frame for target."
            )
            self._h_validmax = 0
            self._w_validmax = 0

        print(
            f"[{self.__class__.__name__}] Target Indices: [0,{self._validtarget_ceilT-1}]. Index={self._validtarget_ceilT-1} has shape [:{self._h_validmax},:{self._w_validmax}].  Available data: {self._data_shape[0]}"
        )

    def get_valid_target_index(self):
        """
        Returns an index which corresponds to a frame which is expected to have a target.
        """

        _, h, w, _ = self._data_shape
        framepixelcount = h * w
        targetpixels = np.array(
            [framepixelcount] * (self._validtarget_ceilT - 1)
            + [self._h_validmax * self._w_validmax]
        )
        targetpixels = targetpixels / np.sum(targetpixels)
        t = np.random.choice(self._validtarget_ceilT, p=targetpixels)
        # t = np.random.randint(0, self._validtarget_ceilT) if self._validtarget_ceilT >= 1 else 0
        h, w = self.get_valid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Valid', index, h,w,t)
        return index

    def get_invalid_target_index(self):
        # if self._validtarget_ceilT == 0:
        # TODO: There may not be enough data for this to work. The better way is to skip using 0 for invalid target.
        # t = np.random.randint(1, self._data_shape[0])
        # elif self._validtarget_ceilT < self._data_shape[0]:
        #     t = np.random.randint(self._validtarget_ceilT, self._data_shape[0])
        # else:
        #     t = self._validtarget_ceilT - 1
        # 5
        # 1.2 => 2
        total_t, h, w, _ = self._data_shape
        framepixelcount = h * w
        available_h = h - self._h_validmax
        if available_h < self._patch_size:
            available_h = 0
        available_w = w - self._w_validmax
        if available_w < self._patch_size:
            available_w = 0

        targetpixels = np.array(
            [available_h * available_w]
            + [framepixelcount] * (total_t - self._validtarget_ceilT)
        )
        t_probab = targetpixels / np.sum(targetpixels)
        t = np.random.choice(
            np.arange(self._validtarget_ceilT - 1, total_t), p=t_probab
        )

        h, w = self.get_invalid_target_hw(t)
        index = self.idx_manager.idx_from_hwt(h, w, t)
        # print('Invalid', index, h,w,t)
        return index

    def get_valid_target_hw(self, t):
        """
        This is the opposite of get_invalid_target_hw. It returns a h,w which is valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(0, self._h_validmax - self._patch_size)
            w = np.random.randint(0, self._w_validmax - self._patch_size)
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def get_invalid_target_hw(self, t):
        """
        This is the opposite of get_valid_target_hw. It returns a h,w which is not valid for target.
        This is only valid for single frame setup.
        """
        if t == self._validtarget_ceilT - 1:
            h = np.random.randint(
                self._h_validmax, self._data_shape[1] - self._patch_size
            )
            w = np.random.randint(
                self._w_validmax, self._data_shape[2] - self._patch_size
            )
        else:
            h = np.random.randint(0, self._data_shape[1] - self._patch_size)
            w = np.random.randint(0, self._data_shape[2] - self._patch_size)
        return h, w

    def _get_tidx(self, index):
        if isinstance(index, int) or isinstance(index, np.int64):
            idx = index
        else:
            idx = index[0]
        return self.idx_manager.get_t(idx)

    def index_should_have_target(self, index):
        tidx = self._get_tidx(index)
        if tidx < self._validtarget_ceilT - 1:
            return True
        elif tidx > self._validtarget_ceilT - 1:
            return False
        else:
            h, w, _ = self.idx_manager.hwt_from_idx(index)
            return (
                h + self._patch_size < self._h_validmax
                and w + self._patch_size < self._w_validmax
            )

    @staticmethod
    def get_reduced_frame_size(data_shape_nhw, fraction):
        n, h, w = data_shape_nhw

        framepixelcount = h * w
        targetpixelcount = int(n * framepixelcount * fraction)

        # We are currently supporting this only when there is just one frame.
        # if np.ceil(pixelcount / framepixelcount) > 1:
        #     return None, None

        lastframepixelcount = targetpixelcount % framepixelcount
        assert data_shape_nhw[1] == data_shape_nhw[2]
        if lastframepixelcount > 0:
            new_size = int(np.sqrt(lastframepixelcount))
            return new_size, new_size
        else:
            assert (
                targetpixelcount / framepixelcount >= 1
            ), "This is not possible in euclidean space :D (so this is a bug)"
            return h, w


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
