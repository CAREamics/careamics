"""Module containing references to the algorithm used in CAREamics."""

__all__ = [
    "N2V2Ref",
    "N2VRef",
    "StructN2VRef",
    "N2VDescription",
    "N2V2Description",
    "StructN2VDescription",
    "StructN2V2Description",
    "N2V",
    "N2V2",
    "STRUCT_N2V",
    "STRUCT_N2V2",
    "CUSTOM",
    "N2N",
    "CARE",
    "CAREDescription",
    "N2NDescription",
    "CARERef",
    "N2NRef",
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
    N2NDescription,
    N2V2Description,
    N2VDescription,
    StructN2V2Description,
    StructN2VDescription,
)
from .references import (
    CARERef,
    N2NRef,
    N2V2Ref,
    N2VRef,
    StructN2VRef,
)
