"""BMZ wrapper for the LVAE model."""

from careamics.models.lvae import LVAE as BaseLVAE


class LVAE(BaseLVAE):
    """BMZ wrapper for the LVAE model.

    This wrapper modifies the forward pass to return
    only the first element of the output tuple from the base LVAE model.

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the base LVAE model.
    **kwargs : dict
        Keyword arguments to pass to the base LVAE model.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the BMZ LVAE wrapper.

        Parameters
        ----------
        *args : tuple
            Positional arguments to pass to the base LVAE model.
        **kwargs : dict
            Keyword arguments to pass to the base LVAE model.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass through the BMZ LVAE wrapper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the LVAE model.

        Returns
        -------
        torch.Tensor
            The first element of the output tuple from the base LVAE model.
        """
        output = super().forward(x)[0]
        if self.predict_logvar:
            # skip the log-var channel
            output = output[:, 0:1]

        return output
