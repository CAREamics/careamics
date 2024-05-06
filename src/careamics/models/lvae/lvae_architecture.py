"""
Ladder VAE (LVAE) Model

The current implementation is based on "Interpretable Unsupervised Diversity Denoising and Artefact Removal, Prakash et al."
"""
import os
import torch 
import torch.nn as nn
from typing import Optional

from .layers import BottomUpDeterministicResBlock



class FirstBottomUpLayer(nn.Module):
    """
    The first block of the Encoder. Its role is to perform a first image compression step.
    It is composed by a sequence of nn.Conv2d + non-linearity + BottomUpDeterministicResBlock (1 or more, default is 1).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[int] = 3,
        # init_stride: Optional[int] = 1,
        num_blocks: Optional[int] = 1,
        activation: Optional[str] = "ReLU",
    ):
        """
        Constructor.

        Parameters
        ----------
        init_stride: int, optional
            The stride used in the initial convolutional block.
        num_blocks: int, optional
            The number of BottomUpDeterministicResBlock used in this layer. 
        """

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            # stride=init_stride
        )

        self.activation = (
            getattr(nn, f"{activation}")() if activation is not None else nn.Identity()
        )

        modules = [self.conv1, self.activation]
        for _ in range(num_blocks):
            modules.append(
                BottomUpDeterministicResBlock(
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    nonlin=nonlin,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    skip_padding=self.encoder_res_block_skip_padding,
                    res_block_kernel=self.encoder_res_block_kernel,
                ))
        
        self.first_bottom_up = nn.Sequential(*modules)


class LVAEEncoder(nn.Module):
    """
    The Encoder of the LVAE model which performs the bottom-up pass.

    NOTES: 
    1. It should return the bu_values
    2. 
    """
    def __init__(self):
        super.__init__()


class LVAEDecoder(nn.Module):
    """
    The Decoder of the LVAE model which perform the top-down pass.
    """

class LVAE(nn.Module):
    """
    The LVAE model.
    """