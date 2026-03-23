---
description: Convenience functions
---



# Convenience functions

As building a full CAREamics configuration requires a complete understanding of the 
various parameters and experience with Pydantic, we provide convenience functions
to create configurations with a only few parameters related to the algorithm users
want to train.

All convenience methods can be found in the `careamics.config` modules. CAREamics 
currently supports [Noise2Void](../../../algorithms/Noise2Void) and its variants, 
[CARE](../../../algorithms/CARE) and [Noise2Noise](../../../algorithms/Noise2Noise). 

``` python title="Import convenience functions"
--8<-- "import_convenience.py:imports"
```

Each method does all the heavy lifting to make the configuration coherent. They share
a certain numbers of mandatory parameters:

- `experiment_name`: The name of the experiment, used to differentiate trained models.
- `data_type`: One of the types supported by CAREamics (`array`, `tiff` or [`custom`](advanced_configuration.md#custom-data-type)).
- `axes`: Axes of the data (e.g. SYX), can only the following letters: `STCZYX`.
- `patch_size`: Size of the patches along the spatial dimensions (e.g. [64, 64]).
- `batch_size`: Batch size to use during training (e.g. 8). This parameter affects the
    memory footprint on the GPU.
- `num_epochs`: Number of epochs.
    
    
Additional optional parameters can be passed to tweak the configuration (e.g. the number
of steps per epoch). 


## Settings the number of steps per epoch

For large images, the number of patches may be very large as well. A consequence are
long epochs, and a sparser sampling in time of the network performances during training.
In such cases, it is useful to set the number of step per epochs. By setting a number
of steps samller than the number of total patches, the epoch length is reduced and
metrics calculation over the validation set occurs more often.


=== "Noise2Void"
    
    ```python title="Configuration with maximum number of steps"
    --8<-- "convenience_functions.py:n2v_steps"
    ```

    1. Set the maximum number of steps. If the number of patches is smaller than that number, then
    it simply trains as if we would not have set the number of steps.

=== "CARE"

    ```python title="Configuration with maximum number of steps"
    --8<-- "convenience_functions.py:care_steps"
    ```

    1. Set the maximum number of steps. If the number of patches is smaller than that number, then
    it simply trains as if we would not have set the number of steps.

=== "Noise2Noise"

    ```python title="Configuration with maximum number of steps"
    --8<-- "convenience_functions.py:n2n_steps"
    ```

    1. Set the maximum number of steps. If the number of patches is smaller than that number, then
    it simply trains as if we would not have set the number of steps.



## Training with channels

When training with multiple channels, the `axes` parameter should contain `C` (e.g. `YXC`).
An error will be then thrown if the optional parameter `n_channels` (or `n_channel_in` for 
CARE and Noise2Noise) is not specified! Likewise if `n_channels` is specified but `C` is not in `axes`.

The correct way is to specify them both at the same time.


=== "Noise2Void"
    
    ```python title="Configuration with multiple channels"
    --8<-- "convenience_functions.py:n2v_channels"
    ```

    1. The axes contain the letter `C`.
    2. The number of channels is specified.

=== "CARE"

    ```python title="Configuration with multiple channels"
    --8<-- "convenience_functions.py:care_channels"
    ```

    1. The axes contain the letter `C`.
    2. The number of channels is specified.
    3. Depending on the CARE task, you also see to set `n_channels_out` (optional).

=== "Noise2Noise"

    ```python title="Configuration with multiple channels"
    --8<-- "convenience_functions.py:n2n_channels"
    ```

    1. The axes contain the letter `C`.
    2. The number of channels is specified.
    3. Depending on the CARE task, you also see to set `n_channels_out` (optional).


!!! warning "Independent channels"
    
    By default, the channels are trained independently: that means that they have
    no influence on each other. As they might have completely different noise
    models, this can lead to better results.

    However, in some cases, you might want to train the channels together to get
    more structural information.


To control whether the channels are trained independently, you can use the 
`independent_channels` parameter:


=== "Noise2Void"
    
    ```python title="Training channels together"
    --8<-- "convenience_functions.py:n2v_mix_channels"
    ```

    1. As previously, we specify the channels in `axes` and `n_channels`.
    2. This ensures that the channels are trained together!

=== "CARE"


    ```python title="Training channels together"
    --8<-- "convenience_functions.py:care_mix_channels"
    ```

    1. As previously, we specify the channels in `axes` and `n_channels_in`.
    2. This ensures that the channels are trained together!


=== "Noise2Noise"

    ```python title="Training channels together"
    --8<-- "convenience_functions.py:n2n_mix_channels"
    ```

    1. As previously, we specify the channels in `axes` and `n_channels`.
    2. This ensures that the channels are trained together!


##  Augmentations

By default CAREamics configuration uses augmentations that are specific to the algorithm
(e.g. Noise2Void) and that are compatible with microscopy images (e.g. flip and 90 degrees
rotations).

### Disable augmentations

However in certain cases, users might want to disable augmentations. For instance if you
have structures that are always oriented in the same direction. To do so there is a single
`augmentations` parameter:

=== "Noise2Void"
    
    ```python title="Configuration without augmentations"
    --8<-- "convenience_functions.py:n2v_no_aug"
    ```

    1. Augmentations are disabled (but normalization and N2V pixel manipulation will still be added
    by CAREamics!).

=== "CARE"

    ```python title="Configuration without augmentations"
    --8<-- "convenience_functions.py:care_no_aug"
    ```

    1. Augmentations are disabled (but normalization will still be added!).

=== "Noise2Noise"

    ```python title="Configuration without augmentations"
    --8<-- "convenience_functions.py:n2n_no_aug"
    ```

    1. Augmentations are disabled (but normalization will still be added!).


### Non-default augmentations

Default augmentations apply a random flip along X or Y and a 90 degrees rotation (note that
there is always for each patch and for each augmentation a 0.5 probability that no augmentation
is applied). For samples that contain objects that are never flipped or rotated (e.g.
objects with always the same orientation, or with patterns along a certain direction), it
will be beneficial to apply non-default augmentations.

For instance, in a case where the objects can only be flipped horizontally, we would
only apply flipping along the `X` axis and not apply any rotation.

=== "Noise2Void"
    
    ```python title="Configuration with non-default augmentations"
    --8<-- "convenience_functions.py:n2v_aug"
    ```

    1. Only flipping along the `X` axis is applied.

=== "CARE"

    ```python title="Configuration with non-default augmentations"
    --8<-- "convenience_functions.py:care_aug"
    ```

    1. Only flipping along the `X` axis is applied.

=== "Noise2Noise"

    ```python title="Configuration with non-default augmentations"
    --8<-- "convenience_functions.py:n2n_aug"
    ```

    1. Only flipping along the `X` axis is applied.


!!! information "Available augmentations"

    The available augmentations are the following:

    - `XYFlipModel`, which can be along `X`, `Y` or both.
    - `XYRandomRotate90Model`


## Choosing a logger

By default, CAREamics simply log the training progress in the console. However, it is 
possible to use either [WandB](https://wandb.ai/site) or [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).

!!! note "Loggers installation"

    Using WandB or TensorBoard require the installation of `extra` dependencies. Check
    out the [installation section](../../../installation/conda_mamba.md#extra-dependencies) to know more about it.


=== "Noise2Void"
    
    ```python title="Configuration with WandB"
    --8<-- "convenience_functions.py:n2v_wandb"
    ```

    1. `wandb` or `tensorboard`

=== "CARE"

    ```python title="Configuration with WandB"
    --8<-- "convenience_functions.py:care_wandb"
    ```

    1. `wandb` or `tensorboard`

=== "Noise2Noise"

    ```python title="Configuration with WandB"
    --8<-- "convenience_functions.py:n2n_wandb"
    ```

    1. `wandb` or `tensorboard`

## (Advanced) Passing data loader parameters

The convenience functions allow passing data loader parameters directly through the
`train_dataloader_params` or `val_dataloader_params` parameters. These are the same parameters as those accepted by the
`torch.utils.data.DataLoader` class (see [PyTorch documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)).


=== "Noise2Void"
    
    ```python title="Configuration with data loader parameters"
    --8<-- "convenience_functions.py:n2v_dataloader_kwargs"
    ```

    1. In practice this is the one parameter you might want to change.
    2. You can also set the parameters for the validation dataloader.

=== "CARE"

    ```python title="Configuration with data loader parameters"
    --8<-- "convenience_functions.py:care_dataloader_kwargs"
    ```

    1. In practice this is the one parameter you might want to change.
    2. You can also set the parameters for the validation dataloader.

=== "Noise2Noise"

    ```python title="Configuration with data loader parameters"
    --8<-- "convenience_functions.py:n2n_dataloader_kwargs"
    ```

    1. In practice this is the one parameter you might want to change.
    2. You can also set the parameters for the validation dataloader.


## (Advanced) Passing model specific parameters

By default, the convenience functions use the default [UNet model parameters](../../../../reference/careamics/config/architectures/unet_model). But if 
you are feeling brave, you can pass model specific parameters in the `model_params` dictionary. 

=== "Noise2Void"
    
    ```python title="Configuration with model specific parameters"
    --8<-- "convenience_functions.py:n2v_model_kwargs"
    ```

    1. The depth of the UNet.
    2. The number of channels in the first layer.
    3. Add any other parameter specific to the model!

=== "CARE"

    ```python title="Configuration with model specific parameters"
    --8<-- "convenience_functions.py:care_model_kwargs"
    ```

    1. The depth of the UNet.
    2. The number of channels in the first layer.
    3. Add any other parameter specific to the model!

=== "Noise2Noise"

    ```python title="Configuration with model specific parameters"
    --8<-- "convenience_functions.py:n2n_model_kwargs"
    ```

    1. The depth of the UNet.
    2. The number of channels in the first layer.
    3. Add any other parameter specific to the model!


!!! note "Model parameters overwriting"

    Some values of the model parameters are not compatible with certain algorithms. 
    Therefore, these are overwritten by the convenience functions. For instance,
    if you pass `in_channels` or `independent_channels` in the `model_kwargs` dictionary, 
    they will be ignored and replaced by the explicit parameters passed to the convenience function.

!!! information "Model parameters"

    The model parameters are the following:

    - `conv_dims`
    - `num_classes`
    - `in_channels`
    - `depth`
    - `num_channels_init`
    - `final_activation`  
    - `n2v2`
    - `independent_channels`

    Description for each parameter can be found in the [code reference](../../../../reference/careamics/config/architectures/unet_model).


## Noise2Void specific parameters

[Noise2Void](../../../algorithms/n2v/) has a few additional parameters that can be set, including for using its 
variants [N2V2](../../../algorithms/n2v2/) and [structN2V](../../../algorithms/structn2v/).

!!! note "Understanding Noise2Void and its variants"

    Before deciding which variant to use, and how to modify the parameters, we recommend
    to die a little a bit on [how each algorithm works](../../../algorithms/index.md)!


### Noise2Void parameters

There are two Noise2Void parameters that influence how the patches are manipulated during
training:

- `roi_size`: This parameter specifies the size of the area used to replace the masked pixel value.
- `masked_pixel_percentage`: This parameter specifies how many pixels per patch will be manipulated.

While the default values are usually fine, they can be tweaked to improve the training
in certain cases.

```python title="Configuration with N2V parameters"
--8<-- "convenience_functions.py:n2v_parameters"
```

### N2V2

To use N2V2, the `use_n2v2` parameter should simply be set to `True`.

```python title="Configuration with N2V2"
--8<-- "convenience_functions.py:n2v2"
```

1. What it does is modifying the architecture of the UNet model and the way the masked
    pixels are replaced.


### structN2V

StructN2V has two parameters that can be set:

- `struct_n2v_axis`: The axis along which the structN2V mask will be applied. By default it
    is set to `none` (structN2V is disabled), you can set it to either `horizontal` or `vertical`.
- `struct_n2v_span`: The size of the structN2V mask.

```python title="Configuration with structN2V"
--8<-- "convenience_functions.py:structn2v"
```


## CARE and Noise2Noise parameters

### Using another loss function

As opposed to Noise2Void, [CARE](../../../algorithms/CARE) and [Noise2Noise](../../../algorithms/Noise2Noise) 
can be trained with different loss functions. This can be set using the `loss` parameter 
(surprise, surprise!).

=== "CARE"
    
    ```python title="Configuration with different loss"
    --8<-- "convenience_functions.py:care_loss"
    ```

    1. `mae` or `mse`

=== "Noise2Noise"

    ```python title="Configuration with different loss"
    --8<-- "convenience_functions.py:n2n_loss"
    ```

    1. `mae` or `mse`

