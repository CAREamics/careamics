from aenum import StrEnum

# python 3.11: https://docs.python.org/3/library/enum.html
class SupportedAlgorithm(StrEnum):
    """Algorithms available in CAREamics.
    
    - N2V: Noise2Void, Krull et al., CVF (2019)
    - N2V2: Noise2Void2, Hoeck et al., ECCV (2022)
    - STRUCTN2V: StructN2V, Broaddus et al., ISBI (ISBI)
    """
    _init_ = 'value __doc__'

    N2V = "n2v", "Noise2Void, an self-supervised algorithm using blind-spot "\
        "training to denoise images."
    N2V2 = "n2v2", "N2V2, an iteration of N2V that removes checkboard artefacts."
    STRUCTN2V = "structn2v", "StructN2V, an iteration of N2V that uses a mask to "\
        "remove horizontal or vertical structured noise."
    # CARE = "care"
    # N2N = "n2n"
    # PN2V = "pn2v"
    # HDN = "hdn"
    # SEG = "segmentation"
