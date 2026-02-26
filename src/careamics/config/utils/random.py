"""Random utilities."""

import random


def generate_random_seed() -> int:
    """Generate a random seed for reproducibility.

    Returns
    -------
    int
        A random integer between 1 and 2^31 - 1.
    """
    return random.randint(1, 2**31 - 1)
