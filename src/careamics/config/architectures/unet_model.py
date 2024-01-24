from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    validator
)


class UNet(BaseModel):
    """
    Pydantic model for a N2V2-compatible UNet.

    The number of filters (base) must be even and minimum 8.

    Attributes
    ----------
    depth : int
        Depth of the model, between 1 and 10 (default 2).
    num_channels_init : int
        Number of filters of the first level of the network, should be even
        and minimum 8 (default 96).
    """

    model_config = ConfigDict(
        use_enum_values=True, protected_namespaces=(), validate_assignment=True
    )
    num_classes: int = Field(default=1, ge=1)
    in_channels: int = Field(default=1, ge=1)
    depth: int = Field(default=2, ge=1, le=10)
    num_channels_init: int = Field(default=32, ge=8, le=1024)
    final_activation: str = Field(default='none', pattern="none|sigmoid|softmax")
    n2v2: bool = Field(default=False)  

    @validator("num_channels_init")
    def validate_num_channels_init(cls, num_channels_init: int) -> int:
        """
        Validate that num_channels_init is even.

        Parameters
        ----------
        num_channels_init : int
            Number of channels.

        Returns
        -------
        int
            Validated number of channels.

        Raises
        ------
        ValueError
            If the number of channels is odd.
        """
        # if odd
        if num_channels_init % 2 != 0:
            raise ValueError(
                f"Number of channels for the bottom layer must be even"
                f" (got {num_channels_init})."
            )

        return num_channels_init