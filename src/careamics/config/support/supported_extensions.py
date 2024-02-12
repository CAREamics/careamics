from enum import Enum

# TODO: change into supported_dataset?
class SupportedExtension(str, Enum):
    """
    Supported extensions for input data.

    Currently supported:
        - tif/tiff: .tiff files.
        - zarr: zarr array.
    """

    TIFF = "tiff"
    TIF = "tif" # TODO do we need both?
    # ZARR = "zarr"

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