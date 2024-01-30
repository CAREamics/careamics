from enum import Enum

# python 3.11: https://docs.python.org/3/library/enum.html
class SupportedAlgorithm(str, Enum):
    """
    Available types of algorithms.

    Currently supported algorithms:
        - n2v
  """

    # CARE = "care"
    N2V = "n2v"
    N2V2 = "n2v2" # TODO to decide whether to remove this
    # N2N = "n2n"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # CUSTOM = "custom"
    # SEGM = "segmentation"
