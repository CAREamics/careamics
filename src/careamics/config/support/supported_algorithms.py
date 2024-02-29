from typing import List

from careamics.utils import BaseEnum


class SupportedAlgorithm(str, BaseEnum):
    """Algorithms available in CAREamics.

    - n2v: a self-supervised algorithm using blind-spot training to denoise 
        images, Krull et al., CVF (2019).
    - n2v2: an iteration of N2V that removes checkboard artefacts, Hoeck et al., 
        ECCV (2022)
    - structn2v: an iteration of N2V that uses a mask to remove horizontal or vertical 
        structured noise, Broaddus et al., ISBI (ISBI).
    - custom: Custom algorithm, allows tuning CAREamics parameters without constraints.
    """


    N2V = "n2v"
    N2V2 = "n2v2"
    STRUCTN2V = "structn2v"
    CUSTOM = "custom"
    # CARE = "care"
    # N2N = "n2n"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # SEG = "segmentation"

    @classmethod
    def get_unsupervised_algorithms(cls) -> List[str]:
        """Return all unsupervised algorithms."""
        return [cls.N2V.value, cls.N2V2.value, cls.STRUCTN2V.value]