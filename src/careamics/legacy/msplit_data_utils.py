"""Legacy train/val/test splitting used in the MicroSplit paper."""

import numpy as np


def _split_in_half(s, e) -> tuple[list[int], list[int]]:
    n = e - s
    s1 = list(np.arange(n // 2))
    s2 = list(np.arange(n // 2, n))
    return [x + s for x in s1], [x + s for x in s2]


def _adjust_for_imbalance_in_fraction_value(
    val: list[int],
    test: list[int],
    val_fraction: float,
    test_fraction: float,
    total_size: int,
) -> tuple[list[int], list[int]]:
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
) -> tuple[list[int], list[int], list[int]]:
    """MicroSplit paper train/val/test sets.

    Parameters
    ----------
    val_fraction : float
        Fraction of validation images.
    test_fraction : float
        Fraction of test images.
    total_size : int
        Total dataset size.
    starting_test : bool, default=False
        Whether to use the first indices for testing, otherwise interleave val and test.

    Returns
    -------
    list[int]
        Train indices.
    list[int]
        Val indices.
    list[int]
        Test indices.
    """
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
            p1, p2 = _split_in_half(s, test_val_size)
            test += p1
            val += p2

    val, test = _adjust_for_imbalance_in_fraction_value(
        val, test, val_fraction, test_fraction, total_size
    )

    return train, val, test
