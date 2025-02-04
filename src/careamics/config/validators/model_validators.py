"""Architecture model validators."""

from careamics.config.architectures import UNetModel


def model_without_n2v2(model: UNetModel) -> UNetModel:
    """Validate that the Unet model does not have the n2v2 attribute.

    Parameters
    ----------
    model : UNetModel
        Model to validate.

    Returns
    -------
    UNetModel
        The validated model.

    Raises
    ------
    ValueError
        If the model has the `n2v2` attribute set to `True`.
    """
    if model.n2v2:
        raise ValueError(
            "The algorithm does not support the `n2v2` attribute in the model. "
            "Set it to `False`."
        )

    return model


def model_without_final_activation(model: UNetModel) -> UNetModel:
    """Validate that the UNet model does not have the final_activation.

    Parameters
    ----------
    model : UNetModel
        Model to validate.

    Returns
    -------
    UNetModel
        The validated model.

    Raises
    ------
    ValueError
        If the model has the final_activation attribute set.
    """
    if model.final_activation != "None":
        raise ValueError(
            "The algorithm does not support a `final_activation` in the model. "
            'Set it to `"None"`.'
        )

    return model


def model_matching_in_out_channels(model: UNetModel) -> UNetModel:
    """Validate that the UNet model has the same number of channel inputs and outputs.

    Parameters
    ----------
    model : UNetModel
        Model to validate.

    Returns
    -------
    UNetModel
        Validated model.

    Raises
    ------
    ValueError
        If the model has different number of input and output channels.
    """
    if model.num_classes != model.in_channels:
        raise ValueError(
            "The algorithm requires the same number of input and output channels. "
            "Make sure that `in_channels` and `num_classes` are equal."
        )

    return model
