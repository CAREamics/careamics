import random

import numpy as np
import torch
from numpy.random import default_rng

GLOBAL_SEED = 42
GLOBAL_RNG = default_rng(seed=GLOBAL_SEED)


def seed_everything(seed: int = GLOBAL_SEED) -> int:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed
