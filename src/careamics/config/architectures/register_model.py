"""Custom model registration utilities."""

from typing import Callable

from torch.nn import Module

CUSTOM_MODELS = {}  # dictionary of custom models {"name": __class__}


def register_model(name: str) -> Callable:
    """Decorator used to register a torch.nn.Module class with a given `name`.

    Parameters
    ----------
    name : str
        Name of the model.

    Returns
    -------
    Callable
        Function allowing to instantiate the wrapped Module class.

    Raises
    ------
    ValueError
        If a model is already registered with that name.

    Examples
    --------
    ```python
    @register_model(name="linear")
    class LinearModel(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.weight = nn.Parameter(ones(in_features, out_features))
            self.bias = nn.Parameter(ones(out_features))

        def forward(self, input):
            return (input @ self.weight) + self.bias
    ```
    """
    if name is None or name == "":
        raise ValueError("Model name cannot be empty.")

    if name in CUSTOM_MODELS:
        raise ValueError(
            f"Model {name} already exists. Choose a different name or run "
            f"`clear_custom_models()` to empty the registry."
        )

    def add_custom_model(model: Module) -> Module:
        """Add a custom model to the registry and return it.

        Parameters
        ----------
        model : Module
            Module class to register.

        Returns
        -------
        Module
            The registered model.
        """
        # add model to the registry
        CUSTOM_MODELS[name] = model

        return model

    return add_custom_model


def get_custom_model(name: str) -> Module:
    """Get the custom model corresponding to `name` from the registry.

    Parameters
    ----------
    name : str
        Name of the model to retrieve.

    Returns
    -------
    Module
        The requested model.

    Raises
    ------
    ValueError
        If the model is not registered.
    """
    if name not in CUSTOM_MODELS:
        raise ValueError(
            f"Model {name} is unknown. Have you registered it using "
            f'@register_model("{name}") as decorator?'
        )

    return CUSTOM_MODELS[name]


def clear_custom_models() -> None:
    """Clear the custom models registry."""
    # clear dictionary
    CUSTOM_MODELS.clear()
