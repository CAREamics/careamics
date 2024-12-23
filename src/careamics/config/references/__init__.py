"""Module containing references to the algorithm used in CAREamics."""

__all__ = [
    "CARE",
    "CUSTOM",
    "N2N",
    "N2V",
    "N2V2",
    "STRUCT_N2V",
    "STRUCT_N2V2",
    "CAREDescription",
    "CARERef",
    "HDNDescription",
    "HDNRef",
    "N2NDescription",
    "N2NRef",
    "N2V2Description",
    "N2V2Ref",
    "N2VDescription",
    "N2VRef",
    "StructN2V2Description",
    "StructN2VDescription",
    "StructN2VRef",
]

from .algorithm_descriptions import (
    CARE,
    CUSTOM,
    N2N,
    N2V,
    N2V2,
    STRUCT_N2V,
    STRUCT_N2V2,
    CAREDescription,
    HDNDescription,
    N2NDescription,
    N2V2Description,
    N2VDescription,
    StructN2V2Description,
    StructN2VDescription,
)
from .references import (
    CARERef,
    HDNRef,
    N2NRef,
    N2V2Ref,
    N2VRef,
    StructN2VRef,
)
