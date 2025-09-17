"""
Ladder VAE (LVAE) Model.

The current implementation is based on "Interpretable Unsupervised Diversity Denoising
and Artefact Removal, Prakash et al."
"""

from collections.abc import Iterable
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from ..activation import get_activation
from .layers import (
    BottomUpDeterministicResBlock,
    BottomUpLayer,
    GateLayer,
    TopDownDeterministicResBlock,
    TopDownLayer,
)
from .utils import Interpolate, ModelType, crop_img_tensor


class LadderVAE(nn.Module):
    """
    Constructor.

    Parameters
    ----------
    input_shape : int
        The size of the input image.
    output_channels : int
        The number of output channels.
    multiscale_count : int
        The number of scales for multiscale processing.
    z_dims : list[int]
        The dimensions of the latent space for each layer.
    encoder_n_filters : int
        The number of filters in the encoder.
    decoder_n_filters : int
        The number of filters in the decoder.
    encoder_conv_strides : list[int]
        The strides for the conv layers encoder.
    decoder_conv_strides : list[int]
        The strides for the conv layers decoder.
    encoder_dropout : float
        The dropout rate for the encoder.
    decoder_dropout : float
        The dropout rate for the decoder.
    nonlinearity : str
        The nonlinearity function to use.
    predict_logvar : bool
        Whether to predict the log variance.
    analytical_kl : bool
        Whether to use analytical KL divergence.

    Raises
    ------
    NotImplementedError
        If only 2D convolutions are supported.
    """

    def __init__(
        self,
        input_shape: int,
        output_channels: int,
        multiscale_count: int,
        z_dims: list[int],
        encoder_n_filters: int,
        decoder_n_filters: int,
        encoder_conv_strides: list[int],
        decoder_conv_strides: list[int],
        encoder_dropout: float,
        decoder_dropout: float,
        nonlinearity: str,
        predict_logvar: bool,
        analytical_kl: bool,
    ):
        super().__init__()

        # -------------------------------------------------------
        # Customizable attributes
        self.image_size = input_shape
        """Input image size. (Z, Y, X) or (Y, X) if the data is 2D."""
        # TODO: we need to be careful with this since used to be an int.
        # the tuple of shapes used to be `self.input_shape`.
        self.target_ch = output_channels
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_strides = decoder_conv_strides
        self._multiscale_count = multiscale_count
        self.z_dims = z_dims
        self.encoder_n_filters = encoder_n_filters
        self.decoder_n_filters = decoder_n_filters
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.nonlin = nonlinearity
        self.predict_logvar = predict_logvar
        self.analytical_kl = analytical_kl
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Model attributes -> Hardcoded
        self.model_type = ModelType.LadderVae  # TODO remove !
        self.encoder_blocks_per_layer = 1
        self.decoder_blocks_per_layer = 1
        self.bottomup_batchnorm = True
        self.topdown_batchnorm = True
        self.topdown_conv2d_bias = True
        self.gated = True
        self.encoder_res_block_kernel = 3
        self.decoder_res_block_kernel = 3
        self.encoder_res_block_skip_padding = False
        self.decoder_res_block_skip_padding = False
        self.merge_type = "residual"
        self.no_initial_downscaling = True
        self.skip_bottomk_buvalues = 0
        self.stochastic_skip = True
        self.learn_top_prior = True
        self.res_block_type = "bacdbacd"  # TODO remove !
        self.mode_pred = False
        self.logvar_lowerbound = -5
        self._var_clip_max = 20
        self._stochastic_use_naive_exponential = False
        self._enable_topdown_normalize_factor = True

        # Attributes that handle LC -> Hardcoded
        self.enable_multiscale = self._multiscale_count > 1
        self.multiscale_retain_spatial_dims = True
        self.multiscale_lowres_separate_branch = False
        self.multiscale_decoder_retain_spatial_dims = (
            self.multiscale_retain_spatial_dims and self.enable_multiscale
        )

        # Derived attributes
        self.n_layers = len(self.z_dims)

        # Others...
        self._tethered_to_input = False
        self._tethered_ch1_scalar = self._tethered_ch2_scalar = None
        if self._tethered_to_input:
            target_ch = 1
            requires_grad = False
            self._tethered_ch1_scalar = nn.Parameter(
                torch.ones(1) * 0.5, requires_grad=requires_grad
            )
            self._tethered_ch2_scalar = nn.Parameter(
                torch.ones(1) * 2.0, requires_grad=requires_grad
            )
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Data attributes
        self.color_ch = 1  # TODO for now we only support 1 channel
        self.normalized_input = True
        # -------------------------------------------------------

        # -------------------------------------------------------
        # Loss attributes
        # enabling reconstruction loss on mixed input
        self.mixed_rec_w = 0
        self.nbr_consistency_w = 0

        # -------------------------------------------------------
        # 3D related stuff
        self._mode_3D = len(self.image_size) == 3  # TODO refac
        self._model_3D_depth = self.image_size[0] if self._mode_3D else 1
        self._decoder_mode_3D = len(self.decoder_conv_strides) == 3
        if self._mode_3D and not self._decoder_mode_3D:
            assert self._model_3D_depth % 2 == 1, "3D model depth should be odd"
        assert (
            self._mode_3D is True or self._decoder_mode_3D is False
        ), "Decoder cannot be 3D when encoder is 2D"
        self._squish3d = self._mode_3D and not self._decoder_mode_3D
        self._3D_squisher = (
            None
            if not self._squish3d
            else nn.ModuleList(
                [
                    GateLayer(
                        channels=self.encoder_n_filters,
                        conv_strides=self.encoder_conv_strides,
                    )
                    for k in range(len(self.z_dims))
                ]
            )
        )
        # TODO: this bit is in the Ashesh's confusing-hacky style... Can we do better?

        # -------------------------------------------------------
        # # Training attributes
        # # can be used to tile the validation predictions
        # self._val_idx_manager = val_idx_manager
        # self._val_frame_creator = None
        # # initialize the learning rate scheduler params.
        # self.lr_scheduler_monitor = self.lr_scheduler_mode = None
        # self._init_lr_scheduler_params(config)
        # self._global_step = 0
        # -------------------------------------------------------

        # -------------------------------------------------------

        # Calculate the downsampling happening in the network
        self.downsample = [1] * self.n_layers
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not self.no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.encoder_blocks_per_layer
        assert len(self.downsample) == self.n_layers
        # -------------------------------------------------------

        # -------------------------------------------------------
        ### CREATE MODEL BLOCKS
        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        self.encoder_conv_op = getattr(nn, f"Conv{len(self.encoder_conv_strides)}d")
        # TODO these should be defined for all layers here ?
        self.decoder_conv_op = getattr(nn, f"Conv{len(self.decoder_conv_strides)}d")
        # TODO: would be more readable to have a derived parameters to use like
        # `conv_dims = len(self.encoder_conv_strides)` and then use `Conv{conv_dims}d`
        stride = 1 if self.no_initial_downscaling else 2
        self.first_bottom_up = self.create_first_bottom_up(stride)

        # Input Branches for Lateral Contextualization
        self.lowres_first_bottom_ups = None
        self._init_multires()

        # Other bottom-up layers
        self.bottom_up_layers = self.create_bottom_up_layers(
            self.multiscale_lowres_separate_branch
        )

        # Top-down layers
        self.top_down_layers = self.create_top_down_layers()
        self.final_top_down = self.create_final_topdown_layer(
            not self.no_initial_downscaling
        )

        # Likelihood module
        # self.likelihood = self.create_likelihood_module()

        # Output layer --> Project to target_ch many channels
        logvar_ch_needed = self.predict_logvar is not None
        self.output_layer = self.parameter_net = self.decoder_conv_op(
            self.decoder_n_filters,
            self.target_ch * (1 + logvar_ch_needed),
            kernel_size=3,
            padding=1,
            bias=self.topdown_conv2d_bias,
        )

        # # gradient norms. updated while training. this is also logged.
        # self.grad_norm_bottom_up = 0.0
        # self.grad_norm_top_down = 0.0
        # PSNR computation on validation.
        # self.label1_psnr = RunningPSNR()
        # self.label2_psnr = RunningPSNR()
        # TODO: did you add this?

        # msg =f'[{self.__class__.__name__}] Stoc:{not self.non_stochastic_version} RecMode:{self.reconstruction_mode} TethInput:{self._tethered_to_input}'
        # msg += f' TargetCh: {self.target_ch}'
        # print(msg)

    ### SET OF METHODS TO CREATE MODEL BLOCKS
    def create_first_bottom_up(
        self,
        init_stride: int,
        num_res_blocks: int = 1,
    ) -> nn.Sequential:
        """
        Method creates the first bottom-up block of the Encoder.

        Its role is to perform a first image compression step.
        It is composed by a sequence of nn.Conv2d + non-linearity +
        BottomUpDeterministicResBlock (1 or more, default is 1).

        Parameters
        ----------
        init_stride: int
            The stride used by the intial Conv2d block.
        num_res_blocks: int, optional
            The number of BottomUpDeterministicResBlocks, default is 1.
        """
        # From what I got from Ashesh, Z should not be touched in any case.
        nonlin = get_activation(self.nonlin)
        conv_block = self.encoder_conv_op(
            in_channels=self.color_ch,
            out_channels=self.encoder_n_filters,
            kernel_size=self.encoder_res_block_kernel,
            padding=(
                0
                if self.encoder_res_block_skip_padding
                else self.encoder_res_block_kernel // 2
            ),
            stride=init_stride,
        )

        modules = [conv_block, nonlin]

        for _ in range(num_res_blocks):
            modules.append(
                BottomUpDeterministicResBlock(
                    conv_strides=self.encoder_conv_strides,
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    nonlin=nonlin,
                    downsample=False,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.encoder_res_block_kernel,
                )
            )

        return nn.Sequential(*modules)

    def create_bottom_up_layers(self, lowres_separate_branch: bool) -> nn.ModuleList:
        """
        Method creates the stack of bottom-up layers of the Encoder.

        that are used to generate the so-called `bu_values`.

        NOTE:
            If `self._multiscale_count < self.n_layers`, then LC is done only in the first
            `self._multiscale_count` bottom-up layers (starting from the bottom).

        Parameters
        ----------
        lowres_separate_branch: bool
            Whether the residual block(s) used for encoding the low-res input are shared
            (`False`) or not (`True`) with the "same-size" residual block(s) in the
            `BottomUpLayer`'s primary flow.
        """
        multiscale_lowres_size_factor = 1
        nonlin = get_activation(self.nonlin)

        bottom_up_layers = nn.ModuleList([])
        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1

            # LC is applied only to the first (_multiscale_count - 1) bottom-up layers
            layer_enable_multiscale = (
                self.enable_multiscale and self._multiscale_count > i + 1
            )

            # This factor determines the factor by which the low-resolution tensor is larger
            # N.B. Only used if layer_enable_multiscale == True, so we updated it only in that case
            multiscale_lowres_size_factor *= 1 + int(layer_enable_multiscale)

            # TODO: check correctness of this
            if self._multiscale_count > 1:
                output_expected_shape = (dim // 2 ** (i + 1) for dim in self.image_size)
            else:
                output_expected_shape = None

            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock), possibly with downsampling between them.
            bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.encoder_blocks_per_layer,
                    n_filters=self.encoder_n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    conv_strides=self.encoder_conv_strides,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.encoder_res_block_kernel,
                    gated=self.gated,
                    lowres_separate_branch=lowres_separate_branch,
                    enable_multiscale=self.enable_multiscale,  # TODO: shouldn't the arg be `layer_enable_multiscale` here?
                    multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
                    multiscale_lowres_size_factor=multiscale_lowres_size_factor,
                    decoder_retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    output_expected_shape=output_expected_shape,
                )
            )

        return bottom_up_layers

    def create_top_down_layers(self) -> nn.ModuleList:
        """
        Method creates the stack of top-down layers of the Decoder.

        In these layer the `bu`_values` from the Encoder are merged with the `p_params` from the previous layer
        of the Decoder to get `q_params`. Then, a stochastic layer generates a sample from the latent distribution
        with parameters `q_params`. Finally, this sample is fed through a TopDownDeterministicResBlock to
        compute the `p_params` for the layer below.

        NOTE 1:
            The algorithm for generative inference approximately works as follows:
                - p_params = output of top-down layer above
                - bu = inferred bottom-up value at this layer
                - q_params = merge(bu, p_params)
                - z = stochastic_layer(q_params)
                - (optional) get and merge skip connection from prev top-down layer
                - top-down deterministic ResNet

        NOTE 2:
            When doing unconditional generation, bu_value is not available. Hence the
            merge layer is not used, and z is sampled directly from p_params.

        """
        top_down_layers = nn.ModuleList([])
        nonlin = get_activation(self.nonlin)
        # NOTE: top-down layers are created starting from the bottom-most
        for i in range(self.n_layers):
            # Check if this is the top layer
            is_top = i == self.n_layers - 1

            if self._enable_topdown_normalize_factor:  # TODO: What is this?
                normalize_latent_factor = (
                    1 / np.sqrt(2 * (1 + i)) if len(self.z_dims) > 4 else 1.0
                )
            else:
                normalize_latent_factor = 1.0

            top_down_layers.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.decoder_blocks_per_layer,
                    n_filters=self.decoder_n_filters,
                    is_top_layer=is_top,
                    conv_strides=self.decoder_conv_strides,
                    upsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=self.merge_type,
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    gated=self.gated,
                    analytical_kl=self.analytical_kl,
                    vanilla_latent_hw=self.get_latent_spatial_size(i),
                    retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    input_image_shape=self.image_size,
                    normalize_latent_factor=normalize_latent_factor,
                    conv2d_bias=self.topdown_conv2d_bias,
                    stochastic_use_naive_exponential=self._stochastic_use_naive_exponential,
                )
            )
        return top_down_layers

    def create_final_topdown_layer(self, upsample: bool) -> nn.Sequential:
        """Create the final top-down layer of the Decoder.

        NOTE: In this layer, (optional) upsampling is performed by bilinear interpolation
        instead of transposed convolution (like in other TD layers).

        Parameters
        ----------
        upsample: bool
            Whether to upsample the input of the final top-down layer
            by bilinear interpolation with `scale_factor=2`.
        """
        # Final top-down layer
        modules = list()

        if upsample:
            modules.append(Interpolate(scale=2))

        for i in range(self.decoder_blocks_per_layer):
            modules.append(
                TopDownDeterministicResBlock(
                    c_in=self.decoder_n_filters,
                    c_out=self.decoder_n_filters,
                    nonlin=get_activation(self.nonlin),
                    conv_strides=self.decoder_conv_strides,
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    gated=self.gated,
                    conv2d_bias=self.topdown_conv2d_bias,
                )
            )
        return nn.Sequential(*modules)

    def _init_multires(self, config=None) -> nn.ModuleList:
        """
        Method defines the input block/branch to encode/compress low-res lateral inputs.

        at different hierarchical levels
        in the multiresolution approach (LC). The role of the input branches is similar
        to the one of the first bottom-up layer in the primary flow of the Encoder,
        namely to compress the lateral input image to a degree that is compatible with
        the one of the primary flow.

        NOTE 1: Each input branch consists of a sequence of Conv2d + non-linearity
        + BottomUpDeterministicResBlock. It is meaningful to observe that the
        `BottomUpDeterministicResBlock` shares the same model attributes with the blocks
        in the primary flow of the Encoder (e.g., c_in, c_out, dropout, etc. etc.).
        Moreover, it does not perform downsampling.

        NOTE 2: `_multiscale_count` attribute defines the total number of inputs to the
        bottom-up pass. In other terms if we have the input patch and n_LC additional
        lateral inputs, we will have a total of (n_LC + 1) inputs.
        """
        stride = 1 if self.no_initial_downscaling else 2
        nonlin = get_activation(self.nonlin)
        if self._multiscale_count is None:
            self._multiscale_count = 1

        msg = (
            f"Multiscale count ({self._multiscale_count}) should not exceed the number"
            f"of bottom up layers ({self.n_layers}) by more than 1.\n"
        )
        assert (
            self._multiscale_count <= 1 or self._multiscale_count <= 1 + self.n_layers
        ), msg  # TODO how ?

        msg = (
            "Multiscale approach only supports monocrome images. "
            f"Found instead color_ch={self.color_ch}."
        )
        # assert self._multiscale_count == 1 or self.color_ch == 1, msg

        lowres_first_bottom_ups = []
        for _ in range(1, self._multiscale_count):
            first_bottom_up = nn.Sequential(
                self.encoder_conv_op(
                    in_channels=self.color_ch,
                    out_channels=self.encoder_n_filters,
                    kernel_size=5,
                    padding="same",
                    stride=stride,
                ),
                nonlin,
                BottomUpDeterministicResBlock(
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    conv_strides=self.encoder_conv_strides,
                    nonlin=nonlin,
                    downsample=False,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                ),
            )
            lowres_first_bottom_ups.append(first_bottom_up)

        self.lowres_first_bottom_ups = (
            nn.ModuleList(lowres_first_bottom_ups)
            if len(lowres_first_bottom_ups)
            else None
        )

    ### SET OF FORWARD-LIKE METHODS
    def bottomup_pass(self, inp: torch.Tensor) -> list[torch.Tensor]:
        """Wrapper of _bottomup_pass()."""
        # TODO Remove wrapper
        return self._bottomup_pass(
            inp,
            self.first_bottom_up,
            self.lowres_first_bottom_ups,
            self.bottom_up_layers,
        )

    def _bottomup_pass(
        self,
        inp: torch.Tensor,
        first_bottom_up: nn.Sequential,
        lowres_first_bottom_ups: nn.ModuleList,
        bottom_up_layers: nn.ModuleList,
    ) -> list[torch.Tensor]:
        """
        Method defines the forward pass through the LVAE Encoder, the so-called.

        Bottom-Up pass.

        Parameters
        ----------
        inp: torch.Tensor
            The input tensor to the bottom-up pass of shape (B, 1+n_LC, H, W), where n_LC
            is the number of lateral low-res inputs used in the LC approach.
            In particular, the first channel corresponds to the input patch, while the
            remaining ones are associated to the lateral low-res inputs.
        first_bottom_up: nn.Sequential
            The module defining the first bottom-up layer of the Encoder.
        lowres_first_bottom_ups: nn.ModuleList
            The list of modules defining Lateral Contextualization.
        bottom_up_layers: nn.ModuleList
            The list of modules defining the stack of bottom-up layers of the Encoder.
        """
        if self._multiscale_count > 1:
            x = first_bottom_up(inp[:, :1])
        else:
            x = first_bottom_up(inp)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need for the top-down pass in bu_values list
        bu_values = []
        for i in range(self.n_layers):
            lowres_x = None
            if self._multiscale_count > 1 and i + 1 < inp.shape[1]:
                lowres_x = lowres_first_bottom_ups[i](inp[:, i + 1 : i + 2])
            x, bu_value = bottom_up_layers[i](x, lowres_x=lowres_x)
            bu_values.append(bu_value)

        return bu_values

    def topdown_pass(
        self,
        bu_values: Union[torch.Tensor, None] = None,
        n_img_prior: Union[torch.Tensor, None] = None,
        constant_layers: Union[Iterable[int], None] = None,
        forced_latent: Union[list[torch.Tensor], None] = None,
        top_down_layers: Union[nn.ModuleList, None] = None,
        final_top_down_layer: Union[nn.Sequential, None] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Method defines the forward pass through the LVAE Decoder, the so-called.

        Top-Down pass.

        Parameters
        ----------
        bu_values: torch.Tensor, optional
            Output of the bottom-up pass. It will have values from multiple layers of
            the ladder.
        n_img_prior: optional
            When `bu_values` is `None`, `n_img_prior` indicates the number of images to
            generate
            from the prior (so bottom-up pass is not used at all here).
        constant_layers: Iterable[int], optional
            A sequence of indexes associated to the layers in which a single instance's
            z is copied over the entire batch (bottom-up path is not used, so only prior
            is used here). Set to `None` to avoid this behaviour.
        forced_latent: list[torch.Tensor], optional
            A list of tensors that are used as fixed latent variables (hence, sampling
            doesn't take place in this case).
        top_down_layers: nn.ModuleList, optional
            A list of top-down layers to use in the top-down pass. If `None`, the method
            uses the default layers defined in the constructor.
        final_top_down_layer: nn.Sequential, optional
            The last top-down layer of the top-down pass. If `None`, the method uses the
            default layers defined in the constructor.
        """
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down

        # Default: no layer is sampled from the distribution's mode
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = (
                "Number of images for top-down generation has to be given "
                "if and only if we're not doing inference"
            )
            raise RuntimeError(msg)
        if inference_mode and prior_experiment:
            msg = (
                "Prior experiments (e.g. sampling from mode) are not"
                " compatible with inference mode"
            )
            raise RuntimeError(msg)

        # Sampled latent variables at each layer
        z = [None] * self.n_layers
        # KL divergence of each layer
        kl = [None] * self.n_layers
        # Kl divergence restricted, only for the LC enabled setup denoiSplit.
        kl_restricted = [None] * self.n_layers
        # mean from which z is sampled.
        q_mu = [None] * self.n_layers
        # log(var) from which z is sampled.
        q_lv = [None] * self.n_layers
        # Spatial map of KL divergence for each layer
        kl_spatial = [None] * self.n_layers
        debug_qvar_max = [None] * self.n_layers
        kl_channelwise = [None] * self.n_layers
        if forced_latent is None:
            forced_latent = [None] * self.n_layers

        # Top-down inference/generation loop
        out = None
        for i in reversed(range(self.n_layers)):
            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            constant_out = i in constant_layers

            # Input for skip connection
            skip_input = out

            # Full top-down layer, including sampling and deterministic part
            out, aux = top_down_layers[i](
                input_=out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
                var_clip_max=self._var_clip_max,
            )
            # Save useful variables
            z[i] = aux["z"]  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux["kl_samplewise"]  # (batch, )
            kl_restricted[i] = aux["kl_samplewise_restricted"]
            kl_spatial[i] = aux["kl_spatial"]  # (batch, h, w)
            q_mu[i] = aux["q_mu"]
            q_lv[i] = aux["q_lv"]

            kl_channelwise[i] = aux["kl_channelwise"]
            debug_qvar_max[i] = aux["qvar_max"]
            # if self.mode_pred is False:
            #     logprob_p += aux['logprob_p'].mean()  # mean over batch
            # else:
            #     logprob_p = None

        # Final top-down layer
        out = final_top_down_layer(out)

        # Store useful variables in a dict to return them
        data = {
            "z": z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            "kl": kl,  # list of tensors with shape (batch, )
            "kl_restricted": kl_restricted,  # list of tensors with shape (batch, )
            "kl_spatial": kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            "kl_channelwise": kl_channelwise,  # list of tensors with shape (batch, ch[i])
            # 'logprob_p': logprob_p,  # scalar, mean over batch
            "q_mu": q_mu,
            "q_lv": q_lv,
            "debug_qvar_max": debug_qvar_max,
        }
        return out, data

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Forward pass through the LVAE model.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (B, C, H, W).
        """
        img_size = x.size()[2:]

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x)
        for i in range(0, self.skip_bottomk_buvalues):
            bu_values[i] = None

        if self._squish3d:
            bu_values = [
                torch.mean(self._3D_squisher[k](bu_value), dim=2)
                for k, bu_value in enumerate(bu_values)
            ]

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        out = self.output_layer(out)

        return out, td_data

    ### SET OF GETTERS
    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, W) or (H, W)
        :return: 2-tuple (H, W)
        """
        # Make size argument into (heigth, width)
        # assert len(size) in [2, 4, 5] # TODO commented out cuz it's weird
        # We're only interested in the Y,X dimensions
        size = size[-2:]

        if self.multiscale_decoder_retain_spatial_dims is True:
            # In this case, we can go much more deeper and so this is not required
            # (in the way it is. ;). More work would be needed if this was to be correctly implemented )
            return list(size)

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = [((s - 1) // dwnsc + 1) * dwnsc for s in size]
        # TODO Needed for pad/crop odd sizes. Move to dataset?
        return padded_size

    def get_latent_spatial_size(self, level_idx: int):
        """Level_idx: 0 is the bottommost layer, the highest resolution one."""
        actual_downsampling = level_idx + 1
        dwnsc = 2**actual_downsampling
        sz = self.get_padded_size(self.image_size)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        assert h == w
        return h

    def get_top_prior_param_shape(self, n_imgs: int = 1):

        # Compute the total downscaling performed in the Encoder
        if self.multiscale_decoder_retain_spatial_dims is False:
            dwnsc = self.overall_downscale_factor
        else:
            # LC allow the encoder latents to keep the same (H, W) size at different levels
            actual_downsampling = self.n_layers + 1 - self._multiscale_count
            dwnsc = 2**actual_downsampling

        h = self.image_size[-2] // dwnsc
        w = self.image_size[-1] // dwnsc
        mu_logvar = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, mu_logvar, h, w)
        # TODO refactor!
        if self._model_3D_depth > 1 and self._decoder_mode_3D is True:
            # TODO check if model_3D_depth is needed ?
            top_layer_shape = (n_imgs, mu_logvar, self._model_3D_depth, h, w)
        return top_layer_shape

    def reset_for_inference(self, tile_size: tuple[int, int] | None = None):
        """Should be called if we want to predict for a different input/output size."""
        self.mode_pred = True
        if tile_size is None:
            tile_size = self.image_size
        self.image_size = tile_size
        for i in range(self.n_layers):
            self.bottom_up_layers[i].output_expected_shape = (
                ts // 2 ** (i + 1) for ts in tile_size
            )
            self.top_down_layers[i].latent_shape = tile_size
