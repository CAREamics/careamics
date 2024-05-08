"""
Ladder VAE (LVAE) Model

The current implementation is based on "Interpretable Unsupervised Diversity Denoising and Artefact Removal, Prakash et al."
"""
import os
from typing import List, Tuple, Dict, Iterable, Optional
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import wandb
from torch.autograd import Variable

### TODO: Replace these imports!!!
from disentangle.analysis.pred_frame_creator import PredFrameCreator
from disentangle.core.data_utils import Interpolate, crop_img_tensor, pad_img_tensor
from disentangle.core.metric_monitor import MetricMonitor
from disentangle.core.psnr import RangeInvariantPsnr
from disentangle.core.sampler_type import SamplerType
from disentangle.loss.exclusive_loss import compute_exclusion_loss
from disentangle.loss.nbr_consistency_loss import NeighborConsistencyLoss
from disentangle.losses import free_bits_kl
from disentangle.metrics.running_psnr import RunningPSNR
from disentangle.nets.noise_model import get_noise_model


from .utils import LossType

from .layers import (
    BottomUpDeterministicResBlock,
    BottomUpLayer,
    TopDownLayer,
    TopDownDeterministicResBlock
) 

from .likelihoods import GaussianLikelihood, NoiseModelLikelihood


class LadderVAE(nn.Module):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2, val_idx_manager=None):
        super().__init__()
        
        # Loss attributes
        self.ch1_recons_w = config.loss.get('ch1_recons_w', 1)
        self.ch2_recons_w = config.loss.get('ch2_recons_w', 1)
        self._restricted_kl = config.loss.get('restricted_kl', False)
        self.kl_loss_formulation = config.loss.get('kl_loss_formulation', None)
        assert self.kl_loss_formulation in [
            None, '', 'usplit', 'denoisplit', 'denoisplit_usplit'], f"""
            Invalid kl_loss_formulation. {self.kl_loss_formulation}"""
        self.kl_annealing = config.loss.kl_annealing
        self.kl_annealtime = self.kl_start = None
        if self.kl_annealing:
            self.kl_annealtime = config.loss.kl_annealtime
            self.kl_start = config.loss.kl_start
        self.loss_type = config.loss.loss_type
        self.kl_weight = config.loss.kl_weight
        self.usplit_kl_weight = config.loss.get('usplit_kl_weight', None)
        self.free_bits = config.loss.free_bits
        self.reconstruction_weight = config.loss.get('reconstruction_weight', 1.0)
        # enabling reconstruction loss on mixed input
        self.mixed_rec_w = 0
        self.mixed_rec_w_step = 0
        self.enable_mixed_rec = False
        self.nbr_consistency_w = 0
        self._exclusion_loss_weight = config.loss.get('exclusion_loss_weight', 0)
        
        self._denoisplit_w = self._usplit_w = None
        if self.loss_type == LossType.DenoiSplitMuSplit:
            self._denoisplit_w = config.loss.denoisplit_w
            self._usplit_w = config.loss.usplit_w
            assert self._denoisplit_w + self._usplit_w == 1

        if self.loss_type in [
                LossType.ElboMixedReconstruction, LossType.ElboSemiSupMixedReconstruction,
                LossType.ElboRestrictedReconstruction
        ]:
            self.mixed_rec_w = config.loss.mixed_rec_weight
            self.mixed_rec_w_step = config.loss.get('mixed_rec_w_step', 0)
            self.enable_mixed_rec = True
            if self.loss_type not in [
                    LossType.ElboSemiSupMixedReconstruction, LossType.ElboMixedReconstruction,
                    LossType.ElboRestrictedReconstruction
            ] and config.data.use_one_mu_std is False:
                raise NotImplementedError(
                    "This cannot work since now, different channels have different mean. One needs to reweigh the "
                    "predicted channels and then take their sum. This would then be equivalent to the input.")
        elif self.loss_type == LossType.ElboWithNbrConsistency:
            self.nbr_consistency_w = config.loss.nbr_consistency_w
            assert 'grid_size' in config.data or 'gridsizes' in config.training
            self._grid_sz = config.data.grid_size if 'grid_size' in config.data else config.data.image_size
            # NeighborConsistencyLoss assumes the batch to be a sequence of [center, left, right, top bottom] images.
            self.nbr_consistency_loss = NeighborConsistencyLoss(
                self._grid_sz,
                nbr_set_count=config.data.get('nbr_set_count', None),
                focus_on_opposite_gradients=config.model.offset_prediction_focus_on_opposite_gradients)
        self.channel_1_w = config.loss.get('channel_1_w', 1)
        self.channel_2_w = config.loss.get('channel_2_w', 1)


        # Training attributes
        self.lr = config.training.lr
        self.lr_scheduler_patience = config.training.lr_scheduler_patience
        # can be used to tile the validation predictions
        self._val_idx_manager = val_idx_manager
        self._val_frame_creator = None
        self._dump_kth_frame_prediction = config.training.get('dump_kth_frame_prediction')
        if self._dump_kth_frame_prediction is not None:
            assert self._val_idx_manager is not None
            dir = os.path.join(config.workdir, 'pred_frames')
            os.mkdir(dir)
            self._dump_epoch_interval = config.training.get('dump_epoch_interval', 1)
            self._val_frame_creator = PredFrameCreator(self._val_idx_manager, self._dump_kth_frame_prediction, dir)    


        # Data attributes
        self._input_is_sum = config.data.input_is_sum
        self.color_ch = config.data.get('color_ch', 1)
        self.img_shape = (config.data.image_size, config.data.image_size)
        # normalized_input: If input is normalized, then we don't normalize the input.
        # We then just normalize the target. Otherwise, both input and target are normalized.
        self.normalized_input = config.data.normalized_input
        self._multiscale_count = config.data.multiscale_lowres_count

        # Model attributes
        self.enable_noise_model = config.model.enable_noise_model
        self._stochastic_use_naive_exponential = config.model.decoder.get('stochastic_use_naive_exponential', False)
        self._enable_topdown_normalize_factor = config.model.get('enable_topdown_normalize_factor', True)
        self.likelihood_gm = self.likelihood_NM = None

        self._tethered_ch1_scalar = self._tethered_ch2_scalar = None
        self._tethered_to_input = config.model.get('tethered_to_input', False)
        if self._tethered_to_input:
            target_ch = 1
            requires_grad = config.model.get('tethered_learnable_scalar', False)
            # a learnable scalar that is multiplied with one channel prediction.
            self._tethered_ch1_scalar = nn.Parameter(torch.ones(1) * 0.5, requires_grad=requires_grad)
            self._tethered_ch2_scalar = nn.Parameter(torch.ones(1) * 2.0, requires_grad=requires_grad)

        # disentangling two grayscale images.
        self.target_ch = target_ch

        self.z_dims = config.model.z_dims
        self.encoder_blocks_per_layer = config.model.encoder.blocks_per_layer
        self.decoder_blocks_per_layer = config.model.decoder.blocks_per_layer

        self.n_layers = len(self.z_dims)
        self.stochastic_skip = config.model.stochastic_skip
        self.bottomup_batchnorm = config.model.encoder.batchnorm
        self.topdown_batchnorm = config.model.decoder.batchnorm

        self.encoder_n_filters = config.model.encoder.n_filters
        self.decoder_n_filters = config.model.decoder.n_filters

        self.encoder_dropout = config.model.encoder.dropout
        self.decoder_dropout = config.model.decoder.dropout
        self.skip_bottomk_buvalues = config.model.get('skip_bottomk_buvalues', 0)

        # whether or not to have bias with Conv2D layer.
        self.topdown_conv2d_bias = config.model.decoder.conv2d_bias

        self.learn_top_prior = config.model.learn_top_prior
        self.res_block_type = config.model.res_block_type
        self.encoder_res_block_kernel = config.model.encoder.res_block_kernel
        self.decoder_res_block_kernel = config.model.decoder.res_block_kernel

        self.encoder_res_block_skip_padding = config.model.encoder.res_block_skip_padding
        self.decoder_res_block_skip_padding = config.model.decoder.res_block_skip_padding

        self.reconstruction_mode = config.model.get('reconstruction_mode', False)

        self.gated = config.model.gated
        if isinstance(data_mean, np.ndarray):
            self.data_mean = torch.Tensor(data_mean)
            self.data_std = torch.Tensor(data_std)
        elif isinstance(data_mean, dict):
            for k in data_mean.keys():
                data_mean[k] = torch.Tensor(data_mean[k]) if not isinstance(data_mean[k], dict) else data_mean[k]
                data_std[k] = torch.Tensor(data_std[k]) if not isinstance(data_std[k], dict) else data_std[k]
            self.data_mean = data_mean
            self.data_std = data_std
        else:
            raise NotImplementedError('data_mean and data_std must be either a numpy array or a dictionary')

        self.noiseModel = get_noise_model(config)
        self.merge_type = config.model.merge_type
        self.analytical_kl = config.model.analytical_kl
        self.no_initial_downscaling = config.model.no_initial_downscaling
        self.mode_pred = config.model.mode_pred
        self.use_uncond_mode_at = use_uncond_mode_at
        self.nonlin = config.model.nonlin

        self.predict_logvar = config.model.predict_logvar
        self.logvar_lowerbound = config.model.logvar_lowerbound
        self.non_stochastic_version = config.model.get('non_stochastic_version', False)
        self._var_clip_max = config.model.var_clip_max

        self.encoder_no_padding_mode = config.model.encoder.res_block_skip_padding is True and config.model.encoder.res_block_kernel > 1
        self.decoder_no_padding_mode = config.model.decoder.res_block_skip_padding is True and config.model.decoder.res_block_kernel > 1

        self.skip_nboundary_pixels_from_loss = config.model.skip_nboundary_pixels_from_loss
        # initialize the learning rate scheduler params.
        self.lr_scheduler_monitor = self.lr_scheduler_mode = None
        self._init_lr_scheduler_params(config)

        self._global_step = 0

        assert (self.data_std is not None)
        assert (self.data_mean is not None)
        
        if self.noiseModel is None:
            self.likelihood_form = "gaussian"
        else:
            self.likelihood_form = "noise_model"

        self.downsample = [1] * self.n_layers

        # Downsample by a factor of 2 at each downsampling operation
        self.overall_downscale_factor = np.power(2, sum(self.downsample))
        if not self.no_initial_downscaling:  # by default do another downscaling
            self.overall_downscale_factor *= 2

        assert max(self.downsample) <= self.encoder_blocks_per_layer
        assert len(self.downsample) == self.n_layers

        # Get class of nonlinear activation from string description
        nonlin = self.get_nonlin()

        
        ### CREATE MODEL BLOCKS
        # First bottom-up layer: change num channels + downsample by factor 2
        # unless we want to prevent this
        stride = 1 if self.no_initial_downscaling else 2
        self.first_bottom_up = self.create_first_bottom_up(stride)
        self.multiscale_retain_spatial_dims = config.model.multiscale_retain_spatial_dims
        self.lowres_first_bottom_ups = self._multiscale_count = None
        self._init_multires(config)

        # other bottom-up layers
        enable_multiscale = self._multiscale_count is not None and self._multiscale_count > 1
        self.multiscale_decoder_retain_spatial_dims = self.multiscale_retain_spatial_dims and enable_multiscale
        self.bottom_up_layers = self.create_bottom_up_layers(config.model.multiscale_lowres_separate_branch)

        # Top-down layers
        self.top_down_layers = self.create_top_down_layers()
        self.final_top_down = self.create_final_topdown_layer(not self.no_initial_downscaling)

        # Likelihood module
        self.likelihood = self.create_likelihood_module()
        
        # Output layer
        logvar_ch_needed = self.predict_logvar is not None
        self.output_layer = self.parameter_net = nn.Conv2d(
            self.decoder_n_filters,
            self.target_ch * (1 + logvar_ch_needed),
            kernel_size=3,
            padding=1,
            bias=self.topdown_conv2d_bias
        )

        # gradient norms. updated while training. this is also logged.
        self.grad_norm_bottom_up = 0.0
        self.grad_norm_top_down = 0.0
        # PSNR computation on validation.
        # self.label1_psnr = RunningPSNR()
        # self.label2_psnr = RunningPSNR()
        self.channels_psnr = [RunningPSNR() for _ in range(target_ch)]
    
        msg =f'[{self.__class__.__name__}] Stoc:{not self.non_stochastic_version} RecMode:{self.reconstruction_mode} TethInput:{self._tethered_to_input}'
        msg += f' TargetCh: {self.target_ch}'
        print(msg)


### SET OF METHODS TO CREATE MODEL BLOCKS
    def create_first_bottom_up(
        self, 
        init_stride: int, 
        num_res_blocks: int = 1,
    ) -> nn.Sequential:
        """
        This method creates the first bottom-up block of the Encoder. 
        Its role is to perform a first image compression step.
        It is composed by a sequence of nn.Conv2d + non-linearity + 
        BottomUpDeterministicResBlock (1 or more, default is 1).
        
        Parameters
        ----------
        init_stride: int
            The stride used by the intial Conv2d block.
        num_res_blocks: int, optional
            The number of BottomUpDeterministicResBlocks to include in the layer, default is 1.
        """
        
        nonlin = self.get_nonlin()
        modules = [
            nn.Conv2d(
                self.color_ch,
                self.encoder_n_filters,
                self.encoder_res_block_kernel,
                padding=0 if self.encoder_res_block_skip_padding else self.encoder_res_block_kernel // 2,
                stride=init_stride
            ),
            nonlin()
        ]
        
        for _ in range(num_res_blocks):
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
                )
            )
            
        return nn.Sequential(*modules)

    def create_bottom_up_layers(
        self, 
        lowres_separate_branch
    ) -> nn.ModuleList:
        """
        This method creates the stack of bottom-up layers of the Encoder
        that are used to generate the so-called `bu_values`.
        
        Parameters
        ----------
        """
        
        # Whether to use Lateral Contextualization (LC)
        multiscale_lowres_size_factor = 1
        enable_multiscale = self._multiscale_count is not None and self._multiscale_count > 1
        
        bottom_up_layers = nn.ModuleList([])
        nonlin = self.get_nonlin()
        for i in range(self.n_layers):
            # Whether this is the top layer
            is_top = i == self.n_layers - 1
            layer_enable_multiscale = enable_multiscale and self._multiscale_count > i + 1
            # if multiscale is enabled, this is the factor by which the lowres tensor will be larger than
            multiscale_lowres_size_factor *= (1 + int(layer_enable_multiscale))
            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            output_expected_shape = (self.img_shape[0] // 2**(i + 1),
                                     self.img_shape[1] // 2**(i + 1)) if self._multiscale_count > 1 else None
            bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.encoder_blocks_per_layer,
                    n_filters=self.encoder_n_filters,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.encoder_res_block_kernel,
                    res_block_skip_padding=self.encoder_res_block_skip_padding,
                    gated=self.gated,
                    lowres_separate_branch=lowres_separate_branch,
                    enable_multiscale=enable_multiscale,
                    multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
                    multiscale_lowres_size_factor=multiscale_lowres_size_factor,
                    decoder_retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    output_expected_shape=output_expected_shape
                    )
                )
        return bottom_up_layers

    def create_top_down_layers(self) -> nn.ModuleList:
        """
        This method creates the stack of top-down layers of the Decoder.
        In these layer the `bu`_values` from the Encoder are merged with 
        the `p_params` from the previous layer of the Decoder to get 
        `q_params`. Then, a sample is generated from the latent distribution
        with parameters `q_params`. Finally, this sample is fed through a 
        TopDownDeterministicResBlock to get the `p_params` for the layer below.
        
        Parameters
        ----------
        
        NOTE:
            The architecture when doing inference is roughly as follows:
               p_params = output of top-down layer above
               bu = inferred bottom-up value at this layer
               q_params = merge(bu, p_params)
               z = stochastic_layer(q_params):
               possibly get skip connection from previous top-down layer
               top-down deterministic ResNet
            
            When doing generation only, the value bu is not available, the
            merge layer is not used, and z is sampled directly from p_params.
            
            Normalization should be applied only with relatively deep networks.
        """
        
        top_down_layers = nn.ModuleList([])
        nonlin = self.get_nonlin()
        for i in range(self.n_layers):
            # Check if this is the top layer
            is_top = i == self.n_layers - 1

            if self._enable_topdown_normalize_factor:
                normalize_latent_factor = 1 / np.sqrt(2 * (1 + i)) if len(self.z_dims) > 4 else 1.0
            else:
                normalize_latent_factor = 1.0

            top_down_layers.append(
                TopDownLayer(
                    z_dim=self.z_dims[i],
                    n_res_blocks=self.decoder_blocks_per_layer,
                    n_filters=self.decoder_n_filters,
                    is_top_layer=is_top,
                    downsampling_steps=self.downsample[i],
                    nonlin=nonlin,
                    merge_type=self.merge_type,
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    stochastic_skip=self.stochastic_skip,
                    learn_top_prior=self.learn_top_prior,
                    top_prior_param_shape=self.get_top_prior_param_shape(),
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    res_block_skip_padding=self.decoder_res_block_skip_padding,
                    gated=self.gated,
                    analytical_kl=self.analytical_kl,
                    restricted_kl=self._restricted_kl,
                    vanilla_latent_hw = self.get_latent_spatial_size(i),
                    # in no_padding_mode, what gets passed from the encoder are not multiples of 2 and so merging operation does not work natively.
                    bottomup_no_padding_mode=self.encoder_no_padding_mode,
                    topdown_no_padding_mode=self.decoder_no_padding_mode,
                    retain_spatial_dims=self.multiscale_decoder_retain_spatial_dims,
                    non_stochastic_version=self.non_stochastic_version,
                    input_image_shape=self.img_shape,
                    normalize_latent_factor=normalize_latent_factor,
                    conv2d_bias=self.topdown_conv2d_bias,
                    stochastic_use_naive_exponential=self._stochastic_use_naive_exponential
                )
            )
        return top_down_layers


    def create_final_topdown_layer(
        self, 
        upsample
    ) -> nn.Sequential:
        """
        This method creates the final top-down layer of the Decoder.
        
        Parameters
        ----------
        
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
                    nonlin=self.get_nonlin(),
                    batchnorm=self.topdown_batchnorm,
                    dropout=self.decoder_dropout,
                    res_block_type=self.res_block_type,
                    res_block_kernel=self.decoder_res_block_kernel,
                    skip_padding=self.decoder_res_block_skip_padding,
                    gated=self.gated,
                    conv2d_bias=self.topdown_conv2d_bias,
                )
            )
        return nn.Sequential(*modules)


    def create_likelihood_module(self):
        """
        This method defines the likelihood module for the current LVAE model.
        The existing likelihood modules are `GaussianLikelihood` and `NoiseModelLikelihood`.
        """
        self.likelihood_gm = GaussianLikelihood(
            self.decoder_n_filters,
            self.target_ch,
            predict_logvar=self.predict_logvar,
            logvar_lowerbound=self.logvar_lowerbound,
            conv2d_bias=self.topdown_conv2d_bias
        )
        self.likelihood_NM = None
        if self.enable_noise_model:
            self.likelihood_NM = NoiseModelLikelihood(
                self.decoder_n_filters, 
                self.target_ch, 
                self.data_mean, 
                self.data_std,
                self.noiseModel
            )
        if self.loss_type == LossType.DenoiSplitMuSplit or self.likelihood_NM is None:
            return self.likelihood_gm
        
        return self.likelihood_NM


    def _init_multires(self, config) -> nn.ModuleList:
        """
        Initialize everything related to multiresolution approach (LC).
        """
        
        stride = 1 if self.no_initial_downscaling else 2
        nonlin = self.get_nonlin()
        if self._multiscale_count is None:
            self._multiscale_count = 1

        msg = "Multiscale count({}) should not exceed the number of bottom up layers ({}) by more than 1"
        msg = msg.format(self._multiscale_count, len(config.model.z_dims))
        assert self._multiscale_count <= 1 or self._multiscale_count <= 1 + len(config.model.z_dims), msg

        msg = "if multiscale is enabled, then we are just working with monocrome images."
        assert self._multiscale_count == 1 or self.color_ch == 1, msg
        lowres_first_bottom_ups = []
        for _ in range(1, self._multiscale_count):
            first_bottom_up = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.color_ch, 
                    out_channels=self.encoder_n_filters, 
                    kernel_size=5,
                    padding=2,
                    stride=stride
                ), 
                nonlin(),
                BottomUpDeterministicResBlock(
                    c_in=self.encoder_n_filters,
                    c_out=self.encoder_n_filters,
                    nonlin=nonlin,
                    batchnorm=self.bottomup_batchnorm,
                    dropout=self.encoder_dropout,
                    res_block_type=self.res_block_type,
                    skip_padding=self.encoder_res_block_skip_padding,
                )
            )
            lowres_first_bottom_ups.append(first_bottom_up)

        self.lowres_first_bottom_ups = nn.ModuleList(lowres_first_bottom_ups) if len(lowres_first_bottom_ups) else None


### SET OF FORWARD-LIKE METHODS
    def bottomup_pass(self, inp: torch.Tensor) -> List[torch.Tensor]:
        """
        Wrapper of _bottomup_pass().
        """
        return self._bottomup_pass(inp, self.first_bottom_up, self.lowres_first_bottom_ups, self.bottom_up_layers)

    def _bottomup_pass(
        self, 
        inp: torch.Tensor, 
        first_bottom_up: nn.Sequential, 
        lowres_first_bottom_ups: nn.ModuleList, 
        bottom_up_layers: nn.ModuleList
    ) -> List[torch.Tensor]:
        """
        This method defines the forward pass throught the LVAE Encoder, the so-called 
        Bottom-Up pass.
        
        Parameters
        ----------
        inp: torch.Tensor
            The input tensor to the bottom-up pass.
        first_bottom_up: nn.Sequential
            The module defining the first bottom-up layer of the Encoder.
        lowres_first_bottom_ups: nn.ModuleList
            The list of modules defining Lateral Contextualization.
        bottom_up_layers: nn.ModuleList
            The list of modules defining the stack of bottom-up layers of the Encoder.
        """
        if self._multiscale_count > 1:
            # Bottom-up initial layer. The first channel is the original input image
            # while additional channels are simply to yield more context.
            x = first_bottom_up(inp[:, :1])
        else:
            x = first_bottom_up(inp)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            lowres_x = None
            if self._multiscale_count > 1 and i + 1 < inp.shape[1]:
                lowres_x = lowres_first_bottom_ups[i](inp[:, i + 1:i + 2])

            x, bu_value = bottom_up_layers[i](x, lowres_x=lowres_x)
            bu_values.append(bu_value)

        return bu_values


    def topdown_pass(
        self,
        bu_values: torch.Tensor = None,
        n_img_prior: torch.Tensor = None,
        mode_layers: Iterable[int] = None,
        constant_layers: Iterable[int] = None,
        forced_latent: List[torch.Tensor] = None,
        top_down_layers: nn.ModuleList = None,
        final_top_down_layer: nn.Sequential = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        This method defines the forward pass throught the LVAE Decoder, the so-called 
        Top-Down pass.
        
        Parameters
        ----------
            bu_values: torch.Tensor, optional
                Output of the bottom-up pass. It will have values from multiple layers of the ladder.
            n_img_prior: optional
                When `bu_values` is `None`, `n_img_prior` indicates the number of images to generate
                from the prior (so bottom-up pass is not used at all here).
            mode_layers: Iterable[int], optional
                A sequence of indexes associated to the layers in which sampling is disabled and 
                the mode (mean value) is used instead. Set to `None` to avoid this behaviour.
            constant_layers: Iterable[int], optional
                A sequence of indexes associated to the layers in which a single instance's z is 
                copied over the entire batch (bottom-up path is not used, so only prior is used here).
                Set to `None` to avoid this behaviour.
            forced_latent: List[torch.Tensor], optional
                A list of tensors that are used as fixed latent variables (hence, sampling doesn't take
                place in this case).
            top_down_layers: nn.ModuleList, optional
                A list of top-down layers to use in the top-down pass. If `None`, the method uses the 
                default layers defined in the contructor.
            final_top_down_layer: nn.Sequential, optional
                The last top-down layer of the top-down pass. If `None`, the method uses the default 
                layers defined in the contructor.
        """
        
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down

        # Default: no layer is sampled from the distribution's mode
        if mode_layers is None:
            mode_layers = []
        if constant_layers is None:
            constant_layers = []
        prior_experiment = len(mode_layers) > 0 or len(constant_layers) > 0

        # If the bottom-up inference values are not given, don't do
        # inference, sample from prior instead
        inference_mode = bu_values is not None

        # Check consistency of arguments
        if inference_mode != (n_img_prior is None):
            msg = ("Number of images for top-down generation has to be given "
                   "if and only if we're not doing inference")
            raise RuntimeError(msg)
        if inference_mode and prior_experiment and (self.non_stochastic_version is False):
            msg = ("Prior experiments (e.g. sampling from mode) are not"
                   " compatible with inference mode")
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

        # log p(z) where z is the sample in the topdown pass
        # logprob_p = 0.

        # Top-down inference/generation loop
        out = out_pre_residual = None
        for i in reversed(range(self.n_layers)):

            # If available, get deterministic node from bottom-up inference
            try:
                bu_value = bu_values[i]
            except TypeError:
                bu_value = None

            # Whether the current layer should be sampled from the mode
            use_mode = i in mode_layers
            constant_out = i in constant_layers
            use_uncond_mode = i in self.use_uncond_mode_at

            # Input for skip connection
            skip_input = out  # TODO or n? or both?

            # Full top-down layer, including sampling and deterministic part
            out, out_pre_residual, aux = top_down_layers[i](
                out,
                skip_connection_input=skip_input,
                inference_mode=inference_mode,
                bu_value=bu_value,
                n_img_prior=n_img_prior,
                use_mode=use_mode,
                force_constant_output=constant_out,
                forced_latent=forced_latent[i],
                mode_pred=self.mode_pred,
                use_uncond_mode=use_uncond_mode,
                var_clip_max=self._var_clip_max
            )
            
            # Save useful variables
            z[i] = aux['z']  # sampled variable at this layer (batch, ch, h, w)
            kl[i] = aux['kl_samplewise']  # (batch, )
            kl_restricted[i] = aux['kl_samplewise_restricted']
            kl_spatial[i] = aux['kl_spatial']  # (batch, h, w)
            q_mu[i] = aux['q_mu']
            q_lv[i] = aux['q_lv']

            kl_channelwise[i] = aux['kl_channelwise']
            debug_qvar_max[i] = aux['qvar_max']
            # if self.mode_pred is False:
            #     logprob_p += aux['logprob_p'].mean()  # mean over batch
            # else:
            #     logprob_p = None
            
        # Final top-down layer
        out = final_top_down_layer(out)

        # Store useful variables in a dict to return them 
        data = {
            'z': z,  # list of tensors with shape (batch, ch[i], h[i], w[i])
            'kl': kl,  # list of tensors with shape (batch, )
            'kl_restricted': kl_restricted, # list of tensors with shape (batch, )
            'kl_spatial': kl_spatial,  # list of tensors w shape (batch, h[i], w[i])
            'kl_channelwise': kl_channelwise,  # list of tensors with shape (batch, ch[i])
            # 'logprob_p': logprob_p,  # scalar, mean over batch
            'q_mu': q_mu,
            'q_lv': q_lv,
            'debug_qvar_max': debug_qvar_max,
        }
        return out, data


    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (B, C, H, W).
        """
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        for i in range(0, self.skip_bottomk_buvalues):
            bu_values[i] = None

        mode_layers = range(self.n_layers) if self.non_stochastic_version else None
        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values, mode_layers=mode_layers)

        if out.shape[-1] > img_size[-1]:
            # Restore original image size
            out = crop_img_tensor(out, img_size)

        out = self.output_layer(out)
        if self._tethered_to_input:
            assert out.shape[1] == 1
            ch2 = self.get_other_channel(out, x_pad)
            out = torch.cat([out, ch2], dim=1)

        return out, td_data
    

# SET OF UTILS METHODS
    def sample_prior(
            self, 
            n_imgs, 
            mode_layers=None, 
            constant_layers=None
        ):

        # Generate from prior
        out, _ = self.topdown_pass(n_img_prior=n_imgs, mode_layers=mode_layers, constant_layers=constant_layers)
        out = crop_img_tensor(out, self.img_shape)

        # Log likelihood and other info (per data point)
        _, likelihood_data = self.likelihood(out, None)

        return likelihood_data['sample']
    
    ### ???
    def sample_from_q(self, x, masks=None):
        """
        This method performs the bottomup_pass() and samples from the 
        obtained distribution.
        """
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(x_pad)
        return self._sample_from_q(bu_values, masks=masks)
    ### ???

    def _sample_from_q(self, bu_values, top_down_layers=None, final_top_down_layer=None, masks=None):
        if top_down_layers is None:
            top_down_layers = self.top_down_layers
        if final_top_down_layer is None:
            final_top_down_layer = self.final_top_down
        if masks is None:
            masks = [None] * len(bu_values)

        msg = "Multiscale is not supported as of now. You need the output from the previous layers to do this."
        assert self.n_layers == 1, msg
        samples = []
        for i in reversed(range(self.n_layers)):
            bu_value = bu_values[i]

            # Note that the first argument can be set to None since we are just dealing with one level
            sample = top_down_layers[i].sample_from_q(None, bu_value, var_clip_max=self._var_clip_max, mask=masks[i])
            samples.append(sample)

        return samples

    def reset_for_different_output_size(self, output_size):
        for i in range(self.n_layers):
            sz = output_size // 2**(1 + i)
            self.bottom_up_layers[i].output_expected_shape = (sz, sz)
            self.top_down_layers[i].latent_shape = (output_size, output_size)


    def pad_input(self, x):
        """
        Pads input x so that its sizes are powers of 2
        :param x:
        :return: Padded tensor
        """
        size = self.get_padded_size(x.size())
        x = pad_img_tensor(x, size)
        return x
    
### SET OF GETTERS 
    def get_nonlin(self):
        nonlin = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
        }
        return nonlin[self.nonlin]


    def get_padded_size(self, size):
        """
        Returns the smallest size (H, W) of the image with actual size given
        as input, such that H and W are powers of 2.
        :param size: input size, tuple either (N, C, H, w) or (H, W)
        :return: 2-tuple (H, W)
        """

        # Make size argument into (heigth, width)
        if len(size) == 4:
            size = size[2:]
        if len(size) != 2:
            msg = ("input size must be either (N, C, H, W) or (H, W), but it "
                   "has length {} (size={})".format(len(size), size))
            raise RuntimeError(msg)

        if self.multiscale_decoder_retain_spatial_dims is True:
            # In this case, we can go much more deeper and so this is not required
            # (in the way it is. ;). More work would be needed if this was to be correctly implemented )
            return list(size)

        # Overall downscale factor from input to top layer (power of 2)
        dwnsc = self.overall_downscale_factor

        # Output smallest powers of 2 that are larger than current sizes
        padded_size = list(((s - 1) // dwnsc + 1) * dwnsc for s in size)

        return padded_size


    def get_latent_spatial_size(self, level_idx):
        """
        level_idx: 0 is the bottommost layer, the highest resolution one.
        """
        actual_downsampling = level_idx + 1
        dwnsc = 2**actual_downsampling
        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        assert h == w
        return h


    def get_top_prior_param_shape(self, n_imgs=1):
        # TODO num channels depends on random variable we're using

        if self.multiscale_decoder_retain_spatial_dims is False:
            dwnsc = self.overall_downscale_factor
        else:
            actual_downsampling = self.n_layers + 1 - self._multiscale_count
            dwnsc = 2**actual_downsampling

        sz = self.get_padded_size(self.img_shape)
        h = sz[0] // dwnsc
        w = sz[1] // dwnsc
        c = self.z_dims[-1] * 2  # mu and logvar
        top_layer_shape = (n_imgs, c, h, w)
        return top_layer_shape


    def get_other_channel(self, ch1, input):
        assert self.data_std['target'].squeeze().shape == (2, )
        assert self.data_mean['target'].squeeze().shape == (2, )
        assert self.target_ch == 2
        ch1_un = ch1[:, :1] * self.data_std['target'][:, :1] + self.data_mean['target'][:, :1]
        input_un = input * self.data_std['input'] + self.data_mean['input']
        ch2_un = self._tethered_ch2_scalar * (input_un - ch1_un * self._tethered_ch1_scalar)
        ch2 = (ch2_un - self.data_mean['target'][:, -1:]) / self.data_std['target'][:, -1:]
        return ch2
    
    
    def get_mixed_prediction(self, prediction, prediction_logvar, data_mean, data_std, channel_weights=None):
        pred_unorm = prediction * data_std['target'] + data_mean['target']
        if channel_weights is None:
            channel_weights = 1

        if self._input_is_sum:
            mixed_prediction = torch.sum(pred_unorm * channel_weights, dim=1, keepdim=True)
        else:
            mixed_prediction = torch.mean(pred_unorm * channel_weights, dim=1, keepdim=True)

        mixed_prediction = (mixed_prediction - data_mean['input'].mean()) / data_std['input'].mean()

        if prediction_logvar is not None:
            if data_std['target'].shape == data_std['input'].shape and torch.all(
                    data_std['target'] == data_std['input']):
                assert channel_weights == 1
                logvar = prediction_logvar
            else:
                var = torch.exp(prediction_logvar)
                var = var * (data_std['target'] / data_std['input'])**2
                if channel_weights != 1:
                    var = var * torch.square(channel_weights)

                # sum of variance.
                mixed_var = 0
                for i in range(var.shape[1]):
                    mixed_var += var[:, i:i + 1]

                logvar = torch.log(mixed_var)
        else:
            logvar = None
        return mixed_prediction, logvar

    def _get_weighted_likelihood(self, ll):
        """
        Each of the channels gets multiplied with a different weight.
        """
        if self.ch1_recons_w == 1 and self.ch2_recons_w == 1:
            return ll
        assert ll.shape[1] == 2, "This function is only for 2 channel images"
        mask1 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
        mask1[:, 0] = 1

        mask2 = torch.zeros((len(ll), ll.shape[1], 1, 1), device=ll.device)
        mask2[:, 1] = 1
        return ll * mask1 * self.ch1_recons_w + ll * mask2 * self.ch2_recons_w