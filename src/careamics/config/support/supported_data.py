"""Data supported by CAREamics."""

from __future__ import annotations

from typing import Union

from careamics.utils import BaseEnum


class SupportedData(str, BaseEnum):
    """Supported data types.

    Attributes
    ----------
    ARRAY : str
        Array data.
    TIFF : str
        TIFF image data.
    CZI : str
        CZI image data.
    CUSTOM : str
        Custom data.
    """

    ARRAY = "array"
    TIFF = "tiff"
    CZI = "czi"
    CUSTOM = "custom"
    # ZARR = "zarr"

    # TODO remove?
    @classmethod
    def _missing_(cls, value: object) -> str:
        """
        Override default behaviour for missing values.

        This method is called when `value` is not found in the enum values. It converts
        `value` to lowercase, removes "." if it is the first character and tries to
        match it with enum values.

        Parameters
        ----------
        value : object
            Value to be matched with enum values.

        Returns
        -------
        str
            Matched enum value.
        """
        if isinstance(value, str):
            lower_value = value.lower()

            if lower_value.startswith("."):
                lower_value = lower_value[1:]

            # attempt to match lowercase value with enum values
            for member in cls:
                if member.value == lower_value:
                    return member

        # still missing
        return super()._missing_(value)

    @classmethod
    def get_extension_pattern(cls, data_type: Union[str, SupportedData]) -> str:
        """
        Get Path.rglob and fnmatch compatible extension.

        Parameters
        ----------
        data_type : SupportedData
            Data type.

        Returns
        -------
        str
            Corresponding extension pattern.
        """
        if data_type == cls.ARRAY:
            raise NotImplementedError(f"Data '{data_type}' is not loaded from a file.")
        elif data_type == cls.TIFF:
            return "*.tif*"
        elif data_type == cls.CZI:
            return "*.czi"
        elif data_type == cls.CUSTOM:
            return "*.*"
        else:
            raise ValueError(f"Data type {data_type} is not supported.")

    @classmethod
    def get_extension(cls, data_type: Union[str, SupportedData]) -> str:
        """
        Get file extension of corresponding data type.

        Parameters
        ----------
        data_type : str or SupportedData
            Data type.

        Returns
        -------
        str
            Corresponding extension.
        """
        if data_type == cls.ARRAY:
            raise NotImplementedError(f"Data '{data_type}' is not loaded from a file.")
        elif data_type == cls.TIFF:
            return ".tiff"
        elif data_type == cls.CZI:
            return ".czi"
        elif data_type == cls.CUSTOM:
            # TODO: improve this message
            raise NotImplementedError("Custom extensions have to be passed elsewhere.")
        else:
            raise ValueError(f"Data type {data_type} is not supported.")
