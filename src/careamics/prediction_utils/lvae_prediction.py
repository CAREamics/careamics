"""Module containing pytorch implementations for obtaining predictions from an LVAE."""

from typing import Optional

import torch

from careamics.models import LVAE

# TODO: convert these functions to lightning module `predict_step`
#   -> mmse_count will have to be an instance attribute?


def lvae_predict_single_sample(
    model: LVAE, input: torch.Tensor
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate a single sample prediction from an LVAE model, for a given input.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    input : torch.tensor
        Input to generate prediction for. Expected shape is (S, C, Y, X).

    Returns
    -------
    tuple of (torch.tensor, optional torch.tensor)
        The first element is the sample prediction, and the second element is the
        log-variance. The log-variance will be None if `model.predict_logvar is None`.
    """
    model.eval()  # Not in original predict code: effects batch_norm and dropout layers
    with torch.no_grad():
        output: torch.Tensor
        output, _ = model(input)  # 2nd item is top-down data dict

    # TODO: what if this is likelihood_NM (Not implemented at the moment)
    # not sure how this will be fully implemented
    #   -> potentially log_var is always None ?
    likelihood_obj = model.likelihood_gm

    sample_prediction: torch.Tensor
    log_var: Optional[torch.Tensor]
    # presently, get_mean_lv just splits the output in 2 if predict_logvar=True,
    #   optionally clips the logvavr if logvar_lowerbound is not None
    sample_prediction, log_var = likelihood_obj.get_mean_lv(output)

    return sample_prediction, log_var


def lvae_predict_mmse(model: LVAE, input: torch.Tensor, mmse_count: int):
    """
    Generate the MMSE (minimum mean squared error) prediction, for a given input.

    This is calculated from the mean of multiple single sample predictions.

    Parameters
    ----------
    model : LVAE
        Trained LVAE model.
    input : torch.tensor
        Input to generate prediction for. Expected shape is (S, C, Y, X).
    mmse_count : int
        Number of samples to generate to calculate MMSE (minimum mean squared error).

    Returns
    -------
    tuple of (torch.tensor, optional torch.tensor)
        The first element is the MMSE prediction, and the second element is the
        log-variance. The log-variance will be None if `model.predict_logvar is None`.
    """
    input_shape = input.shape
    output_shape = (input_shape[0], model.target_ch, *input_shape[2:])
    log_var: Optional[torch.Tensor]
    # pre-declare empty array to fill with individual sample predictions
    sample_predictions = torch.zeros(size=(mmse_count, *output_shape))
    for mmse_idx in range(mmse_count):
        sample_prediction, lv = lvae_predict_single_sample(model, input)
        # only keep the log variance of the first sample prediction
        # TODO: confirm
        if mmse_idx == 0:
            log_var = lv

        # store sample predictions
        sample_predictions[mmse_idx, ...] = sample_prediction

    # take the mean of the sample predictions
    mmse_prediction = torch.mean(sample_prediction, dim=0, keepdim=True)
    return mmse_prediction, log_var
