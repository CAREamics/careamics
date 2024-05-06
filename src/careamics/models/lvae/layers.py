"""
Script containing the common layers (nn.Module) reused by the LadderVAE architecture.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from copy import deepcopy
from typing import Union, Tuple



class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.
    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = batch norm
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (batchnorm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(self,
                 channels: int,
                 nonlin,
                 kernel=None,
                 groups=1,
                 batchnorm: bool = True,
                 block_type: str = None,
                 dropout=None,
                 gated=None,
                 skip_padding=False,
                 conv2d_bias=True):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError("kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        self.skip_padding = skip_padding
        pad = [0] * len(kernel) if self.skip_padding else [k // 2 for k in kernel]
        print(kernel, pad)
        self.gated = gated
        modules = []

        if block_type == 'cabdcabd':
            for i in range(2):
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(nonlin())
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                if dropout is not None:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbacd':
            for i in range(2):
                if batchnorm:
                    modules.append(nn.BatchNorm2d(channels))
                modules.append(nonlin())
                conv = nn.Conv2d(channels, channels, kernel[i], padding=pad[i], groups=groups, bias=conv2d_bias)
                modules.append(conv)
                modules.append(nn.Dropout2d(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        # if gated:
        #     modules.append(GateLayer2d(channels, 1, nonlin))
        # self.block = nn.Sequential(*modules)

    def forward(self, x):

        out = self.block(x)
        if out.shape != x.shape:
            return out + F.center_crop(x, out.shape[-2:])
        else:
            return out + x


class ResBlockWithResampling(nn.Module):
    """
    Residual block that takes care of resampling steps (each by a factor of 2).
    The mode can be top-down or bottom-up, and the block does up- and
    down-sampling by a factor of 2, respectively. Resampling is performed at
    the beginning of the block, through strided convolution.
    The number of channels is adjusted at the beginning and end of the block,
    through convolutional layers with kernel size 1. The number of internal
    channels is by default the same as the number of output channels, but
    min_inner_channels overrides this behaviour.
    Other parameters: kernel size, nonlinearity, and groups of the internal
    residual block; whether batch normalization and dropout are performed;
    whether the residual path has a gate layer at the end. There are a few
    residual block structures to choose from.
    """

    def __init__(
        self,
        mode,
        c_in,
        c_out,
        nonlin=nn.LeakyReLU,
        resample=False,
        res_block_kernel=None,
        groups=1,
        batchnorm=True,
        res_block_type=None,
        dropout=None,
        min_inner_channels=None,
        gated=None,
        lowres_input=False,
        skip_padding=False,
        conv2d_bias=True
    ):
        super().__init__()
        assert mode in ['top-down', 'bottom-up']
        if min_inner_channels is None:
            min_inner_channels = 0
        inner_filters = max(c_out, min_inner_channels)

        # Define first conv layer to change channels and/or up/downsample
        if resample:
            if mode == 'bottom-up':  # downsample
                self.pre_conv = nn.Conv2d(in_channels=c_in,
                                          out_channels=inner_filters,
                                          kernel_size=3,
                                          padding=1,
                                          stride=2,
                                          groups=groups,
                                          bias=conv2d_bias)
            elif mode == 'top-down':  # upsample
                self.pre_conv = nn.ConvTranspose2d(in_channels=c_in,
                                                   out_channels=inner_filters,
                                                   kernel_size=3,
                                                   padding=1,
                                                   stride=2,
                                                   groups=groups,
                                                   output_padding=1,
                                                   bias=conv2d_bias)
        elif c_in != inner_filters:
            self.pre_conv = nn.Conv2d(c_in, inner_filters, 1, groups=groups, bias=conv2d_bias)
        else:
            self.pre_conv = None

        # Residual block
        self.res = ResidualBlock(
            channels=inner_filters,
            nonlin=nonlin,
            kernel=res_block_kernel,
            groups=groups,
            batchnorm=batchnorm,
            dropout=dropout,
            gated=gated,
            block_type=res_block_type,
            skip_padding=skip_padding,
            conv2d_bias=conv2d_bias,
        )
        # Define last conv layer to get correct num output channels
        if inner_filters != c_out:
            self.post_conv = nn.Conv2d(inner_filters, c_out, 1, groups=groups, bias=conv2d_bias)
        else:
            self.post_conv = None

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)

        x = self.res(x)
        if self.post_conv is not None:
            x = self.post_conv(x)
        return x


class TopDownDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, upsample=False, **kwargs):
        kwargs['resample'] = upsample
        super().__init__('top-down', *args, **kwargs)


class BottomUpDeterministicResBlock(ResBlockWithResampling):

    def __init__(self, *args, downsample=False, **kwargs):
        kwargs['resample'] = downsample
        super().__init__('bottom-up', *args, **kwargs)

class BottomUpLayer(nn.Module):
    """
    Bottom-up deterministic layer for inference, roughly the same as the
    small deterministic Resnet in top-down layers. Consists of a sequence of
    bottom-up deterministic residual blocks with downsampling.
    """

    def __init__(
        self,
        n_filters: int,
        n_res_blocks: int,
        downsampling_steps: int = 0,
        nonlin=None,
        batchnorm: bool = True,
        dropout: Union[None, float] = None,
        res_block_type: str = None,
        res_block_kernel: int = None,
        res_block_skip_padding: bool = False,
        gated: bool = None,
        multiscale_lowres_size_factor: int = None,
        enable_multiscale: bool = False,
        lowres_separate_branch=False,
        multiscale_retain_spatial_dims: bool = False,
        decoder_retain_spatial_dims: bool = False,
        output_expected_shape=None
    ):
        """
        Args:
            n_res_blocks: Number of BottomUpDeterministicResBlock blocks present in this layer.
            n_filters:      Number of channels which is present through out this layer.
            downsampling_steps: How many times downsampling has to be done in this layer. This is typically 1.
            nonlin: What non linear activation is to be applied at various places in this module.
            batchnorm: Whether to apply batch normalization at various places or not.
            dropout: Amount of dropout to be applied at various places.
            res_block_type: Example: 'bacdbac'. It has the constitution of the residual block.
            gated: This is also an argument for the residual block. At the end of residual block, whether 
            there should be a gate or not.
            res_block_kernel:int => kernel size for the residual blocks in the bottom up layer.
            multiscale_lowres_size_factor: How small is the bu_value when compared with low resolution tensor.
            enable_multiscale: Whether to enable multiscale or not.
            multiscale_retain_spatial_dims: typically the output of the bottom-up layer scales down spatially.
                                            However, with this set, we return the same spatially sized tensor.
            output_expected_shape: What should be the shape of the output of this layer. Only used if enable_multiscale is True.
        """
        super().__init__()
        self.enable_multiscale = enable_multiscale
        self.lowres_separate_branch = lowres_separate_branch
        self.multiscale_retain_spatial_dims = multiscale_retain_spatial_dims
        self.output_expected_shape = output_expected_shape
        self.decoder_retain_spatial_dims = decoder_retain_spatial_dims
        assert self.output_expected_shape is None or self.enable_multiscale is True

        bu_blocks_downsized = []
        bu_blocks_samesize = []
        for _ in range(n_res_blocks):
            do_resample = False
            if downsampling_steps > 0:
                do_resample = True
                downsampling_steps -= 1
            block = BottomUpDeterministicResBlock(
                c_in=n_filters,
                c_out=n_filters,
                nonlin=nonlin,
                downsample=do_resample,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                skip_padding=res_block_skip_padding,
                gated=gated,
            )
            if do_resample:
                bu_blocks_downsized.append(block)
            else:
                bu_blocks_samesize.append(block)

        self.net_downsized = nn.Sequential(*bu_blocks_downsized)
        self.net = nn.Sequential(*bu_blocks_samesize)
        # using the same net for the lowresolution (and larger sized image)
        self.lowres_net = self.lowres_merge = self.multiscale_lowres_size_factor = None
        if self.enable_multiscale:
            self._init_multiscale(
                n_filters=n_filters,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
                multiscale_lowres_size_factor=multiscale_lowres_size_factor,
            )

        msg = f'[{self.__class__.__name__}] McEnabled:{int(enable_multiscale)} '
        if enable_multiscale:
            msg += f'McParallelBeam:{int(multiscale_retain_spatial_dims)} McFactor{multiscale_lowres_size_factor}'
        print(msg)

    def _init_multiscale(self,
                         n_filters=None,
                         nonlin=None,
                         batchnorm=None,
                         dropout=None,
                         res_block_type=None,
                         multiscale_retain_spatial_dims=None,
                         multiscale_lowres_size_factor=None):
        self.multiscale_lowres_size_factor = multiscale_lowres_size_factor
        self.lowres_net = self.net
        if self.lowres_separate_branch:
            self.lowres_net = deepcopy(self.net)

        self.lowres_merge = MergeLowRes(
            channels=n_filters,
            merge_type='residual',
            nonlin=nonlin,
            batchnorm=batchnorm,
            dropout=dropout,
            res_block_type=res_block_type,
            multiscale_retain_spatial_dims=multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=self.multiscale_lowres_size_factor,
        )

    def forward(self, x, lowres_x=None):
        primary_flow = self.net_downsized(x)
        primary_flow = self.net(primary_flow)

        if self.enable_multiscale is False:
            assert lowres_x is None
            return primary_flow, primary_flow

        if lowres_x is not None:
            lowres_flow = self.lowres_net(lowres_x)
            merged = self.lowres_merge(primary_flow, lowres_flow)
        else:
            merged = primary_flow

        if self.multiscale_retain_spatial_dims is False or self.decoder_retain_spatial_dims is True:
            return merged, merged

        if self.output_expected_shape is not None:
            expected_shape = self.output_expected_shape
        else:
            fac = self.multiscale_lowres_size_factor
            expected_shape = (merged.shape[-2] // fac, merged.shape[-1] // fac)
            assert merged.shape[-2:] != expected_shape

        value_to_use_in_topdown = crop_img_tensor(merged, expected_shape)
        return merged, value_to_use_in_topdown


class TopDownLayer(nn.Module):
    """
    Top-down layer, including stochastic sampling, KL computation, and small
    deterministic ResNet with upsampling.
    The architecture when doing inference is roughly as follows:
       p_params = output of top-down layer above
       bu = inferred bottom-up value at this layer
       q_params = merge(bu, p_params)
       z = stochastic_layer(q_params)
       possibly get skip connection from previous top-down layer
       top-down deterministic ResNet
    When doing generation only, the value bu is not available, the
    merge layer is not used, and z is sampled directly from p_params.
    If this is the top layer, at inference time, the uppermost bottom-up value
    is used directly as q_params, and p_params are defined in this layer
    (while they are usually taken from the previous layer), and can be learned.
    """

    def __init__(self,
                 z_dim: int,
                 n_res_blocks: int,
                 n_filters: int,
                 is_top_layer: bool = False,
                 downsampling_steps: int = None,
                 nonlin=None,
                 merge_type: str = None,
                 batchnorm: bool = True,
                 dropout: Union[None, float] = None,
                 stochastic_skip: bool = False,
                 res_block_type=None,
                 res_block_kernel=None,
                 res_block_skip_padding=None,
                 groups: int = 1,
                 gated=None,
                 learn_top_prior=False,
                 top_prior_param_shape=None,
                 analytical_kl=False,
                 bottomup_no_padding_mode=False,
                 topdown_no_padding_mode=False,
                 retain_spatial_dims: bool = False,
                 restricted_kl=False,
                 vanilla_latent_hw: int = None,
                 non_stochastic_version=False,
                 input_image_shape: Union[None, Tuple[int, int]] = None,
                 normalize_latent_factor=1.0,
                 conv2d_bias: bool = True,
                 stochastic_use_naive_exponential=False):
        """
            Args:
                z_dim:          This is the dimension of the latent space.
                n_res_blocks:   Number of TopDownDeterministicResBlock blocks
                n_filters:      Number of channels which is present through out this layer.
                is_top_layer:   Whether it is top layer or not.
                downsampling_steps: How many times upsampling has to be done in this layer. This is typically 1.
                nonlin: What non linear activation is to be applied at various places in this module.
                merge_type: In Top down layer, one merges the information passed from q() and upper layers.
                            This specifies how to mix these two tensors.
                batchnorm: Whether to apply batch normalization at various places or not.
                dropout: Amount of dropout to be applied at various places.
                stochastic_skip: Previous layer's output is mixed with this layer's stochastic output. So, 
                                the previous layer's output has a way to reach this level without going
                                through the stochastic process. However, technically, this is not a skip as
                                both are merged together. 
                res_block_type: Example: 'bacdbac'. It has the constitution of the residual block.
                gated: This is also an argument for the residual block. At the end of residual block, whether 
                        there should be a gate or not.
                learn_top_prior: Whether we want to learn the top prior or not. If set to False, for the top-most
                                 layer, p will be N(0,1). Otherwise, we will still have a normal distribution. It is 
                                 just that the mean and the stdev will be different.
                top_prior_param_shape: This is the shape of the tensor which would contain the mean and the variance
                                        of the prior (which is normal distribution) for the top most layer.
                analytical_kl:  If True, typical KL divergence is calculated. Otherwise, an approximate of it is 
                            calculated.
                retain_spatial_dims: If True, the the latent space of encoder remains at image_shape spatial resolution for each topdown layer. What this means for one topdown layer is that the input spatial size remains the output spatial size.
                            To achieve this, we centercrop the intermediate representation.
                input_image_shape: This is the shape of the input patch. when retain_spatial_dims is set to True, then this is used to ensure that the output of this layer has this shape. 
                normalize_latent_factor: Divide the latent space (q_params) by this factor.
                conv2d_bias:    Whether or not bias should be present in the Conv2D layer.
        """

        super().__init__()

        self.is_top_layer = is_top_layer
        self.z_dim = z_dim
        self.stochastic_skip = stochastic_skip
        self.learn_top_prior = learn_top_prior
        self.analytical_kl = analytical_kl
        self.bottomup_no_padding_mode = bottomup_no_padding_mode
        self.topdown_no_padding_mode = topdown_no_padding_mode
        self.retain_spatial_dims = retain_spatial_dims
        self.latent_shape = input_image_shape if self.retain_spatial_dims else None
        self.non_stochastic_version = non_stochastic_version
        self.normalize_latent_factor = normalize_latent_factor
        self._vanilla_latent_hw = vanilla_latent_hw
        
        # Define top layer prior parameters, possibly learnable
        if is_top_layer:
            self.top_prior_params = nn.Parameter(torch.zeros(top_prior_param_shape), requires_grad=learn_top_prior)

        # Downsampling steps left to do in this layer
        dws_left = downsampling_steps

        # Define deterministic top-down block: sequence of deterministic
        # residual blocks with downsampling when needed.
        block_list = []

        for _ in range(n_res_blocks):
            do_resample = False
            if dws_left > 0:
                do_resample = True
                dws_left -= 1
            block_list.append(
                TopDownDeterministicResBlock(
                    n_filters,
                    n_filters,
                    nonlin,
                    upsample=do_resample,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    res_block_kernel=res_block_kernel,
                    skip_padding=res_block_skip_padding,
                    gated=gated,
                    conv2d_bias=conv2d_bias,
                    groups=groups,
                ))
        self.deterministic_block = nn.Sequential(*block_list)

        # Define stochastic block with 2d convolutions
        if self.non_stochastic_version:
            self.stochastic = NonStochasticBlock2d(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                transform_p_params=(not is_top_layer),
                groups=groups,
                conv2d_bias=conv2d_bias,
            )
        else:
            self.stochastic = NormalStochasticBlock2d(
                c_in=n_filters,
                c_vars=z_dim,
                c_out=n_filters,
                transform_p_params=(not is_top_layer),
                vanilla_latent_hw=vanilla_latent_hw,
                restricted_kl=restricted_kl,
                use_naive_exponential=stochastic_use_naive_exponential,
            )

        if not is_top_layer:

            # Merge layer, combine bottom-up inference with top-down
            # generative to give posterior parameters
            self.merge = MergeLayer(
                channels=n_filters,
                merge_type=merge_type,
                nonlin=nonlin,
                batchnorm=batchnorm,
                dropout=dropout,
                res_block_type=res_block_type,
                res_block_kernel=res_block_kernel,
                conv2d_bias=conv2d_bias,
            )

            # Skip connection that goes around the stochastic top-down layer
            if stochastic_skip:
                self.skip_connection_merger = SkipConnectionMerger(
                    channels=n_filters,
                    nonlin=nonlin,
                    batchnorm=batchnorm,
                    dropout=dropout,
                    res_block_type=res_block_type,
                    merge_type=merge_type,
                    conv2d_bias=conv2d_bias,
                    res_block_kernel=res_block_kernel,
                    res_block_skip_padding=res_block_skip_padding,
                )
        print(f'[{self.__class__.__name__}] normalize_latent_factor:{self.normalize_latent_factor}')

    def sample_from_q(self, input_, bu_value, var_clip_max=None, mask=None):
        """
        We sample from q
        """
        if self.is_top_layer:
            q_params = bu_value
        else:
            # NOTE: Here the assumption is that the vampprior is only applied on the top layer.
            n_img_prior = None
            p_params = self.get_p_params(input_, n_img_prior)
            q_params = self.merge(bu_value, p_params)

        sample = self.stochastic.sample_from_q(q_params, var_clip_max)
        if mask:
            return sample[mask]
        return sample

    def get_p_params(self, input_, n_img_prior):
        p_params = None
        # If top layer, define parameters of prior p(z_L)
        if self.is_top_layer:
            p_params = self.top_prior_params

            # Sample specific number of images by expanding the prior
            if n_img_prior is not None:
                p_params = p_params.expand(n_img_prior, -1, -1, -1)

        # Else the input from the layer above is the prior parameters
        else:
            p_params = input_

        return p_params

    def align_pparams_buvalue(self, p_params, bu_value):
        """
        In case the padding is not used either (or both) in encoder and decoder, we could have a mismatch. Doing a centercrop to ensure that both remain aligned.
        """
        if bu_value.shape[-2:] != p_params.shape[-2:]:
            assert self.bottomup_no_padding_mode is True
            if self.topdown_no_padding_mode is False:
                assert bu_value.shape[-1] > p_params.shape[-1]
                bu_value = F.center_crop(bu_value, p_params.shape[-2:])
            else:
                if bu_value.shape[-1] > p_params.shape[-1]:
                    bu_value = F.center_crop(bu_value, p_params.shape[-2:])
                else:
                    p_params = F.center_crop(p_params, bu_value.shape[-2:])
        return p_params, bu_value

    def forward(self,
                input_: Union[None, torch.Tensor] = None,
                skip_connection_input=None,
                inference_mode=False,
                bu_value=None,
                n_img_prior=None,
                forced_latent: Union[None, torch.Tensor] = None,
                use_mode: bool = False,
                force_constant_output=False,
                mode_pred=False,
                use_uncond_mode=False,
                var_clip_max: Union[None, float] = None):
        """
        Args:
            input_: output from previous top_down layer.
            skip_connection_input: Currently, this is output from the previous top down layer. 
                                It is mixed with the output of the stochastic layer.
            inference_mode: In inference mode, q_params is not None. Otherwise it is. When q_params is None,
                            everything is generated from the p_params. So, the encoder is not used at all.
            bu_value: Output of the bottom-up pass layer of the same level as this top-down.
            n_img_prior: This affects just the top most top-down layer. This is only present if inference_mode=False.
            forced_latent: If this is a tensor, then in stochastic layer, we don't sample by using p() & q(). We simply 
                            use this as the latent space sampling.
            use_mode:      If it is true, we still don't sample from the q(). We simply 
                            use the mean of the distribution as the latent space.
            force_constant_output: This ensures that only the first sample of the batch is used. Typically used 
                                when infernce_mode is False
            mode_pred: If True, then only prediction happens. Otherwise, KL divergence loss also gets computed.
            use_uncond_mode: Used only when mode_pred=True
            var_clip_max: This is the maximum value the log of the variance of the latent vector for any layer can reach.
        """
        # Check consistency of arguments
        inputs_none = input_ is None and skip_connection_input is None
        if self.is_top_layer and not inputs_none:
            raise ValueError("In top layer, inputs should be None")

        p_params = self.get_p_params(input_, n_img_prior)

        # In inference mode, get parameters of q from inference path,
        # merging with top-down path if it's not the top layer
        if inference_mode:
            if self.is_top_layer:
                q_params = bu_value
                if mode_pred is False:
                    p_params, bu_value = self.align_pparams_buvalue(p_params, bu_value)
            else:
                if use_uncond_mode:
                    q_params = p_params
                else:
                    p_params, bu_value = self.align_pparams_buvalue(p_params, bu_value)
                    q_params = self.merge(bu_value, p_params)

        # In generative mode, q is not used
        else:
            q_params = None

        # Sample from either q(z_i | z_{i+1}, x) or p(z_i | z_{i+1})
        # depending on whether q_params is None

        # This is done, purely for stablity. See Very deep VAEs generalize autoregressive models.
        if self.normalize_latent_factor:
            q_params = q_params / self.normalize_latent_factor

        x, data_stoch = self.stochastic(p_params=p_params,
                                        q_params=q_params,
                                        forced_latent=forced_latent,
                                        use_mode=use_mode,
                                        force_constant_output=force_constant_output,
                                        analytical_kl=self.analytical_kl,
                                        mode_pred=mode_pred,
                                        use_uncond_mode=use_uncond_mode,
                                        var_clip_max=var_clip_max)

        # Skip connection from previous layer
        if self.stochastic_skip and not self.is_top_layer:
            if self.topdown_no_padding_mode is True:
                # the output of last TopDown layer was of size 64*64. Due to lack of padding, currecnt x has become, say 60*60.
                skip_connection_input = F.center_crop(skip_connection_input, x.shape[-2:])

            x = self.skip_connection_merger(x, skip_connection_input)

        # Save activation before residual block: could be the skip
        # connection input in the next layer
        x_pre_residual = x
        if self.retain_spatial_dims:
            # when we don't want to do padding in topdown as well, we need to spare some boundary pixels which would be used up.
            extra_len = (self.topdown_no_padding_mode is True) * 3

            # # this means that the x should be of the same size as config.data.image_size. So, we have to centercrop by a factor of 2 at this point.
            # assert x.shape[-1] >= self.latent_shape[-1] // 2 + extra_len
            # we assume that one topdown layer will have exactly one upscaling layer.
            new_latent_shape = (self.latent_shape[0] // 2 + extra_len, self.latent_shape[1] // 2 + extra_len)

            # If the LC is not applied on all layers, then this can happen.
            if x.shape[-1] > new_latent_shape[-1]:
                x = F.center_crop(x, new_latent_shape)

        # Last top-down block (sequence of residual blocks)
        x = self.deterministic_block(x)

        if self.topdown_no_padding_mode:
            x = F.center_crop(x, self.latent_shape)

        keys = [
            'z',
            'kl_samplewise',
            'kl_samplewise_restricted',
            'kl_spatial',
            'kl_channelwise',
            # 'logprob_p',
            'logprob_q',
            'qvar_max'
        ]
        data = {k: data_stoch.get(k, None) for k in keys}
        data['q_mu'] = None
        data['q_lv'] = None
        if data_stoch['q_params'] is not None:
            q_mu, q_lv = data_stoch['q_params']
            data['q_mu'] = q_mu
            data['q_lv'] = q_lv
        return x, x_pre_residual, data