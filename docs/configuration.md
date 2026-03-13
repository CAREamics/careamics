---
icon: lucide/sliders-horizontal
description: Configuring CAREamics
---

# Configuring CAREamics

To start with CAREamics, we need to create a configuration object that holds most of the
useful parameters. The configuration ensures cross-validation and coherence of the
parameters, in particular avoiding sets of parameters that could trigger errors 
deep in the library.

Configuration can be created using any of the algorithm-specific convenience functions
below. We provide a simple function with a minimum set of parameters, and an advanced 
function giving access to many more.

```python
--8<-- "configuration_simple.py:all_configs"
```

!!! note "CARE and Noise2Noise"

    CARE and Noise2Noise configurations have the exact same set of parameters, contrary
    to Noise2Void. In this section, we only show the CARE configuration, but the same
    applies to Noise2Noise by simply swapping `create_care_config` with
    `create_n2n_config`.


## Simple configuration

The simple configuration functions are designed to only expose the parameters most
commonly used. This is a good starting point for most experiments.

=== "Noise2Void"
    
    ```python title="Configure Noise2Void"
    --8<-- "configuration_simple.py:config_n2v"
    ```
 
    1. The length of the patch size is conditioned on the presence of the `Z` axis.

=== "CARE/N2N"

    ```python title="Configure CARE"
    --8<-- "configuration_simple.py:config_care"
    ```

    1. The length of the patch size is conditioned on the presence of the `Z` axis.

- `experiment_name`: The experiment name is used in the logging and automatic model
saving. It should only contain letters, numbers, underscores, dashes and spaces.
- `data_type`: The data type impacts other parameters and which features may be available.
CAREamics supports `array` (when passing `numpy` arrays directly), `tiff`, `zarr`, `czi`
and `custom`. Refer to the [data section](./data.md) for more details.
- `axes`: The axes of the data, in the order they have on disk (or in memory). This is
important to identify correctly the spatial and channel dimensions. Refer to the
[data section](./data.md) for tips on how to identify axes.
- `patch_size`: The size of the patches to extract from the data during training. Note
that the patch size only refers to spatial axes (`X`, `Y` and optionally `Z`). Patch sizes
are power of 2 and are greater than 8. They are also usually the same for `X` and `Y` axes.
`patch_size` should be 2D for axes without `Z`, and 3D for axes with `Z`.
- `batch_size`: The number of patches to use in each training batch.
- `num_epochs`: The number of epochs to train for. Note that in the case of large
datasets, you might want to also set the [number of steps parameter](#reducing-the-number-of-steps).



!!! note "Training with `T` as depth axis"

    If you want to use your `T` axis as the depth axis, simply relabel it as `Z`. Note
    that this is not compatible with `data_type="czi"`, see [data section](./data.md).


### Reducing the number of steps


Each training epoch cycles through all patches. Therefore, for large datasets, an epoch
can be lengthy and validation happening rarely. In this case, it is useful to set the
number of steps `num_steps`.

=== "Noise2Void"
    
    ```python title="Setting the number of steps"
    --8<-- "configuration_simple.py:config_n2v_steps"
    ```

    1. Use a number smaller than the total number of steps (given the batch size), see
    notes below.

=== "CARE/N2N"

    ```python title="Setting the number of steps"
    --8<-- "configuration_simple.py:config_care_steps"
    ```

    1. Use a number smaller than the total number of steps (given the batch size), see
    notes below.


!!! note "How many steps per epoch?"

    Each epoch consists of `n_patchs / batch_size` steps. The total number of steps is
    shown in the console during training:

    ```sh
    Epoch 1: 12%|█████████████                                     | 36/300
    ```

    While there is a programmaticaly way to know how many patches would CAREamics
    extract from the data, it is easier to simply run a training for an epoch and
    check the console output.



!!! note "Advanced `num_epochs` and `num_steps`"

    The `num_epochs` and `num_steps` correspond to the `max_epochs` and
    `limit_train_batches` parameters of the Pytorch Lightning `Trainer`. Refer to the
    [Trainer API](https://lightning.ai/docs/pytorch/stable/common/trainer.html#init) for
    details about these parameters.


### Augmentations

CAREamics applies augmentations to the training patches, by default random flips in X or
Y, and random rotations by 90 degrees. In certain cases, these augmentations may not be
desirable, for example when the result of the augmentation is not a possible occurence
in the data. In microscopy, this can happen when there are structures that have always
the same orientation, or noise with a spatial correlation. To set them,
use the `augmentations` parameter.

=== "Noise2Void"
    
    ```python title="Setting augmentations"
    --8<-- "configuration_simple.py:config_n2v_augmentations"
    ```

    1. These are all the possible occurences.

=== "CARE/N2N"

    ```python title="Setting augmentations"
    --8<-- "configuration_simple.py:config_care_augmentations"
    ```

    1. These are all the possible occurences.


To disable augmentations, set `augmentations=[]`.


!!! note "How are augmentations applied?"

    Each augmentation has a 0.5 of being applied to each patch. The random 90 degree
    rotations applies either a 90, 180 or 270 rotations (if applied). The augmentations
    are applied sequentially, such that a patch can be flipped in X, not flipped in Y,
    and then rotated by 180 degrees.


### Channels

Channels are a particular type of axes, and they influence the way the deep-learning
model is build. As a result, when `C` is present in the `axes`, additional parameters
need to be set depending on the algorithm.


=== "Noise2Void"
    
    ```python title="Setting channels parameters"
    --8<-- "configuration_simple.py:config_n2v_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. For Noise2Void, the number of input and output channels are the same, so we only
    need to set `n_channels`.

=== "CARE/N2N"

    ```python title="Setting channels parameters"
    --8<-- "configuration_simple.py:config_care_channels"
    ```

    1. Channels are considered to be present as soon as `C` is in `axes`.
    2. For CARE/N2N, the number of input and output channels can be different, so we
    need to set `n_channels_in` and optionally `n_channels_out`.

    Note that if `n_channels_out` is not set, it will be set to the same value as
    `n_channels_in`.

!!! note "Advanced channels parameters"

    The advanced CAREamics configuration gives access to more channel related parameters,
    such as sub-setting or channel independence during training. Refer to the
    [advanced configuration](#advanced-configuration) section for more details.


### Validation

When no validation data is provided, CAREamics will automatically split some patches
from the training data to use as validation. The number of validation patches is set 
then governed by the `num_val_patches` parameter. By default, it is set to `8`.

=== "Noise2Void"
    
    ```python title="Setting the number of validation patches"
    --8<-- "configuration_simple.py:config_n2v_val"
    ```

    1. Choose an appropriate number of validation patches, depending on the size of the
    training data, to avoid pulling too many patches from the training data.

=== "CARE/N2N"

    ```python title="Setting the number of validation patches"
    --8<-- "configuration_simple.py:config_care_val"
    ```

    1. Choose an appropriate number of validation patches, depending on the size of the
    training data, to avoid pulling too many patches from the training data, while
    maintaining meaningful validation.


!!! note "What happens when validation data is passed?"

    In the presence of validation data, the `num_val_patches` parameter is ignored and
    the effective number of validation patches is determined by the size of the
    validation data.

    You can however limit the number of validation steps using PyTorch Lightning 
    parameters, refer to the [advanced training parameters](#pytorch-and-lightning-parameters) section.


## Advanced configuration

More parameters are available by using the advanced configuration convenience functions.
In this section, we explore these additional parameters.

### Training in memory

Where the training data resides influences the speed at which patches can be extracted,
and in turn total training time. The faster way to train is to hold all the data in
memory. However, this is only possible when the data is small enough to fit in the RAM.
Data can be loaded in memory by setting the `in_memory` parameter to `True` in the
configuration.


=== "Noise2Void"
    
    ```python title="Setting in-memory training"
    --8<-- "configuration_advanced.py:adv_config_n2v_in_memory"
    ```

    1. Only `array`, `tiff` and `custom` are compatible with in-memory training.

=== "CARE/N2N"

    ```python title="Setting in-memory training"
    --8<-- "configuration_advanced.py:adv_config_care_in_memory"
    ```

    1. Only `array`, `tiff` and `custom` are compatible with in-memory training.


!!! note "`data_type` and `in_memory` parameters"

    Only `tiff` and `custom` are compatible with `in_memory=True`. For `array`, this is
    automatically set to `True` and cannot be set to `False`. For `czi` and `zarr`,
    training is done by using random access to the data on disk and currently in-memory
    is not implemented.

    For more details on `custom` data type, refer to the [data](./data.md) section.


### Subsetting channels

When the data has channels, it is possible to train from a subset of them only by
passing list of channel indices to the `channels` parameter.


=== "Noise2Void"
    
    ```python title="Selecting a subset of channels"
    --8<-- "configuration_advanced.py:adv_config_n2v_subchannels"
    ```

    1. For channels to be considered present, `C` needs to be in `axes`.
    2. Training would only be performed using two channels, the first and third, since 
    channels are indexed starting from 0.

=== "CARE/N2N"

    ```python title="Selecting a subset of channels"
    --8<-- "configuration_advanced.py:adv_config_care_subchannels"
    ```

    1. For channels to be considered present, `C` needs to be in `axes`.
    2. Training would only be performed using two channels, the first and third, since 
    channels are indexed starting from 0.


!!! note "Number of channels"

    In these examples, you might notice that `n_channels`/`n_channels_in` are not set,
    although they are required when `C` is in `axes`. The reason is that when `channels`
    is set, the number of channels is automatically inferred from `channels`. 

    In the case of CARE/N2N, if `n_channels_out` is also set automatically to the size
    of `channels`, but can also be set to a different value.


### Channel independence

By default, channels are trained independently. This means that the channels do not inform each other during training. For algorith such as Noise2Void, this may be desirable as the noise is uncorrelated between channels. By default, `independent_channels` is set to `True`. To disable channel independence, set `independent_channels=False`.

=== "Noise2Void"
    
    ```python title="Training non-independent channels"
    --8<-- "configuration_advanced.py:adv_config_n2v_ind_channels"
    ```

    1. `C` must be in `axes`.
    2. We need to specify the number of channels.
    3. Set the channels to inform each other.

=== "CARE/N2N"

    ```python title="Training non-independent channels"
    --8<-- "configuration_advanced.py:adv_config_care_ind_channels"
    ```


    1. `C` must be in `axes`.
    2. We need to specify the number of channels. You may also set `n_channels_out` to a different value if you want the output channels to be different from the input channels.
    3. Set the channels to inform each other.

!!! note "What does channel independence mean?"

    In effect, training the channels independently is equivalent to training a separate model for each channel.

### Normalization

CAREamics offers various normalization methods that can be set using the `normalization`
and `normalization_params` parameters. The normalization is applied to any patch or
image before applying the model, therefore it is applied in both training and prediction.
For more details on the available normalization methods and their parameters, refer to the
[Code reference]() section.


The various normalizations are the following:

| Method | Optional parameters|
|----------|----------|
| `"mean_std"` | `input_means`, `input_stds`,`target_means`, `target_stds`, `per_channel=True`|
| `"quantile"` | `lower_quantile=[0.01]`, `upper_quantile=[0.99]` , `per_channel=True`, `input_lower_quantile_values`, `input_upper_quantile_values`, `target_lower_quantile_values`, `target_upper_quantile_values`|
| `"minmax"`    | `input_mins`, `input_maxes` , `target_mins`, `target_maxes`, `per_channel=True`|
| `"none"` | No parameters |

!!! note "Noise2Void and targets"

    While normalization methods have parameters for the targets, there is no target in Noise2Void and these parameters will not be used.


The `quantile` normalization has two parameters, `lower_quantile` and `upper_quantile`, that are set by default to `0.01` and `0.99` respectively. These parameters can be changed by passing them in `normalization_params` as dictionary. By default, `mean_std` normalization is applied, which correspond to zero-mean, unit-variance normalization. To disable normalization, set `normalization="none"`

=== "Noise2Void"
    
    ```python title="Specifying the normalization method"
    --8<-- "configuration_advanced.py:adv_config_n2v_norm"
    ```

    1. The normalization method is choosen by passing `normalization`.
    2. It is possible to change default parameters (here for `quantile`). Alternatively, pre-computed. normalization parameters can be passed here.

=== "CARE/N2N"

    ```python title="Specifying the normalization method"
    --8<-- "configuration_advanced.py:adv_config_care_norm"
    ```

    1. The normalization method is choosen by passing `normalization`.
    2. It is possible to change default parameters (here for `quantile`). Alternatively, pre-computed. normalization parameters can be passed here.


!!! note "Passing pre-computed normalization parameters"

    If the normalization parameters (e.g. `input_means` and `input_stds`) are `None` in the configuration, then the dataset will compute them over the entire training dataset. 

    However, if you have pre-computed normalization parameters, you can pass them in the configuration using the `normalization_params` parameter. This can save time during training, as the dataset will not need to compute them.


=== "Noise2Void"
    
    ```python title="Passing pre-computed parameters"
    --8<-- "configuration_advanced.py:adv_config_n2v_norm_params"
    ```

    1. Here we are selecting the zero-mean unit-variance normalization, which is the default.
    2. We pass pre-computed parameters.

=== "CARE/N2N"

    ```python title="Passing pre-computed parameters"
    --8<-- "configuration_advanced.py:adv_config_care_norm_params"
    ```

    1. Here we are selecting the zero-mean unit-variance normalization, which is the default.
    2. We pass pre-computed parameters.
    3. In CARE and N2N, we can also pass pre-computed target statistics. For N2N they should actually be the same as the input ones.


In the presence of channels, all parameters that are passed to `normalization_params` should have a value for each channel, unless `per_channel` is set to `False`. By default, `per_channel` is set to `True`.


=== "Noise2Void"
    
    ```python title="Setting normalization in the presence of channels"
    --8<-- "configuration_advanced.py:adv_config_n2v_norm_ch"
    ```

    1. We have channels.
    2. We need to set the number of channels.
    3. Here we have to pass as many values as there are channels for each parameter.
    4. Note that if we set `per_channel` to `False`, a single value is expected in the other parameters.


=== "CARE/N2N"

    ```python title="Setting normalization in the presence of channels"
    --8<-- "configuration_advanced.py:adv_config_care_norm_ch"
    ```

    1. We have channels.
    2. We need to set the number of channels.
    3. Here we have to pass as many values as there are channels for each parameter.
    4. Note that if we set `per_channel` to `False`, a single value is expected in the other parameters.

### Choosing a logger

By default, CAREamics uses the CSV logger from PyTorch Lightning, saving all the loss and metrics to a csv file. In addition, we can use more advanced logging tools, such as [WandB](https://wandb.ai/) or [Tensorboard](https://www.tensorflow.org/tensorboard). To use these loggers, simply set the `logger` parameter to `"wandb"` or `"tensorboard"`.



=== "Noise2Void"
    
    ```python title="Choosing a logger"
    --8<-- "configuration_advanced.py:adv_config_n2v_logger"
    ```

    1. We choose WandB in addition to the CSV logger.

=== "CARE/N2N"

    ```python title="Choosing a logger"
    --8<-- "configuration_advanced.py:adv_config_care_logger"
    ```

    1. We choose WandB in addition to the CSV logger.


### Number of workers

The `num_workers` parameter controls the number of workers used to load the data during training. It can be used to optimize data loading performance.


=== "Noise2Void"
    
    ```python title="Setting the number of workers"
    --8<-- "configuration_advanced.py:adv_config_n2v_num_workers"
    ```

    1. We set the number of workers for data loading.

=== "CARE/N2N"

    ```python title="Setting the number of workers"
    --8<-- "configuration_advanced.py:adv_config_care_num_workers"
    ```

    1. We set the number of workers for data loading.


!!! note "Which value to choose?"

    A general rule of thumb is to set the number of workers to the number of CPU cores available. `num_workers=0` means that the data loading will be done in the main process, which can be a bottleneck but can also be necessary in certain environments (e.g. Windows without the possibility to [use multi-processing](https://docs.pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection)).


### Setting a seed

Setting a seed allows fixing the series of random choices happening during training, and would allow to reproduce the same training run. To set a seed, simply set the `seed` parameter to an integer value.


=== "Noise2Void"
    
    ```python title="Setting the seed"
    --8<-- "configuration_advanced.py:adv_config_n2v_seed"
    ```

=== "CARE/N2N"

    ```python title="Setting the seed"
    --8<-- "configuration_advanced.py:adv_config_care_seed"
    ```


## Noise2Void flavours and parameters

Noise2Void have additional parameters that in some cases can make a difference. Furthermore, N2V2 and structN2V are two compatible variants of Noise2Void that are adressing particular short-comings and limitations. 


### Running N2V2

N2V2 is a variant of Noise2Void that mitigates checkerboard artefacts that arise in Noise2Void (e.g. in the presence of salt and pepper noise, or hot pixels). N2V2 can be enabled by simply setting `n2v2=True` in the configuration.

```python title="Using N2V2"
--8<-- "configuration_n2v.py:config_n2v2"
```


### structN2V

structN2V is a variant of Noise2Void that is designed to better handle structured noise, such as line artefacts. It does so by masking out a line of pixels instead of a single pixel during training. To use structN2V, set the `struct_n2v_axis` and `struct_n2v_span` parameters.

```python title="Using structN2V"
--8<-- "configuration_n2v.py:adv_config_structn2v"
```

1. Choices are `horizontal` or `vertical`.
2. The number of pixels to mask out on each side of the pixel masked by Noise2Void.


### Noise2Void parameters

Noise2Void has two parameters that can be set in the configuration: `roi_size` and `masked_pixel_percentage`. A good understanding of the Noise2Void algorithm is necessary to understand the effect of these parameters. Refer to the [Noise2Void algorithm]() section for more details.

```python title="Using Noise2Void parameters"
--8<-- "configuration_n2v.py:adv_config_n2v_params"
```

1. The roi size is the region around the masked pixels from which replacement values are pulled.
2. The percentage of pixels to mask in each patch during training.


## PyTorch and Lightning Parameters

Since CAREamics uses PyTorch Lightning under the hood, many of the parameters that can be used to tune the behaviour of the training can be set though our API.


### Model parameters

While not directly Lighning parameters, the parameters of the UNet model used by CAREamics can be passed to the configuration via `model_params`. Refer to the [Code reference]() for more details about which parameters are available.

=== "Noise2Void"
    
    ```python title="Passing `Trainer` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_trainer"
    ```

=== "CARE/N2N"

    ```python title="Passing `Trainer` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_trainer"
    ```

Some parameters of the model are set automatically based on the parameters given to the configuration. Here is a list of parameters that can only be set via `model_params`:
- `depth`: number of levels in the UNet, by default `2`.
- `num_channels_init`: number of convolutional filters in the first layer of the UNet, by default `32`.
- `residual`: whether to add a residual connection from the input to the output, by default `False`.
- `use_batch_norm`: whether to use batch normalization in the model.


### Trainer parameters

The trainer parameters allow setting complex behaviours during training, from when to stop to advanced gradient manipulation. Refer to the [Trainer](https://lightning.ai/docs/pytorch/stable/common/trainer.html#init) documentation for a list of parameters, their meaning and accepted values.

=== "Noise2Void"
    
    ```python title="Passing `Trainer` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_trainer"
    ```

    1. Pass parameters as a dictionary to `trainer_params`.

=== "CARE/N2N"

    ```python title="Passing `Trainer` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_trainer"
    ```

    1. Pass parameters as a dictionary to `trainer_params`.


!!! note "`max_epochs` and `limit_train_batches` parameters"

    The CAREamics configuration defines `num_epochs` and `num_steps` parameters that are passed to the `Trainer` as `max_epochs` and `limit_train_batches`. If `max_epochs` and `limit_train_batches` are passed to `trainer_params`, their values will be overwritten by `num_epochs` and `num_steps`.

### Optimizer parameters

The optimizer governs how the model parameters are updated during training. By default, CAREamics uses the `Adam` optimizer. To change the optimizer or its parameters, use the `optimizer` and `optimizer_params` parameters in the configuration. Refer to the [PyTorch optimizer](https://docs.pytorch.org/docs/stable/optim.html#algorithms) page for a list of optimizers and their parameters.


=== "Noise2Void"
    
    ```python title="Passing `Optimizer` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_opt"
    ```

    1. Choose the optimizer.
    2. Pass parameters as a dictionary to `optimizer_params`.

=== "CARE/N2N"

    ```python title="Passing `Optimizer` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_opt"
    ```

    1. Choose the optimizer.
    2. Pass parameters as a dictionary to `optimizer_params`.


!!! note "Supported optimizers"

    Note that CAREamics currently only support `Adam`, `SGD` and `Adamax`.


### Learning rate schedulers

Learning rate schedulers allow changing the learning rate during training, which can be useful to improve training performance. To use a learning rate scheduler, set the `lr_scheduler` and `lr_scheduler_params` parameters in the configuration. Refer to the [PyTorch learning rate scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) page for a list of learning rate schedulers and their parameters.

=== "Noise2Void"
    
    ```python title="Passing `LRScheduler` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_lr"
    ```

    1. Choose the learning rate scheduler.
    2. Pass parameters as a dictionary to `lr_scheduler_params`.

=== "CARE/N2N"

    ```python title="Passing `LRScheduler` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_lr"
    ```

    1. Choose the learning rate scheduler.
    2. Pass parameters as a dictionary to `lr_scheduler_params`.


!!! note "Supported learning rate schedulers"

    Note that CAREamics currently only support `ReduceLROnPlateau` and `StepLR`.


### Dataloader parameters

PyTorch dataloaders have various parameters that can be set to optimize data loading performance. To set these parameters, use the `train_dataloader_params` or `val_dataloader_params` parameters in the configuration. Refer to the [PyTorch dataloader](https://docs.pytorch.org/docs/stable/data.html) page for a list of dataloader parameters and their meaning.

=== "Noise2Void"
    
    ```python title="Passing `Dataloader` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_train_dm"
    ```

=== "CARE/N2N"

    ```python title="Passing `Dataloader` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_train_dm"
    ```

!!! note "`shuffle` in `train_dataloader_params`"

    If passing `train_dataloader_params`, then `shuffle` needs to be present. That is not the case for the validation dataloader. The reason is that CAREamics automatically uses `shuffle=True` (the advised setting) for the training dataloader. If you wish to override the CAREamics training dataloader parameters, you need to specify which `shuffle` value you desire.


### Checkpoint parameters

A checkpoint is the state of the training and of the model at a particular time point during training. In particular, the final model is also saved as a checkpoint. There are several [Lightning checkpoints parameters](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html) that govern the behavior of the checkpointing. 

If you want to overried the CAREamics defaults, set `checkpoint_params`.

=== "Noise2Void"
    
    ```python title="Passing `Checkpoint` parameters"
    --8<-- "configuration_lightning.py:adv_config_n2v_checkpoint"
    ```

=== "CARE/N2N"

    ```python title="Passing `Checkpoint` parameters"
    --8<-- "configuration_lightning.py:adv_config_care_checkpoint"
    ```

### Noise2Void without validation

Since validation is not strictly necessary for Noise2Void, it is possible to train
without validation data and without automatic splitting of the training data, thus
making the overall training faster. To do so, a few parameters need to be set in the
configuration.

```python title="Training Noise2Void without validation"
--8<-- "configuration_lightning.py:adv_config_n2v_no_val"
```

1. We tell the trian/val splitting module to split `0` validation patches.
2. We set the monitoring of the learning rate scheduler to `train_loss_epoch`.
3. Finally, we disable the validation step in PyTorch Lightning.

!!! note "Removing validation in other algorithms"

    Do not remove validation for CARE! In supervised training, validation is critical
    to assess whether the network has trained meaningfully.


## Saving and loading

Configurations are automatically saved with the checkpoints, but we can nonetheless
manually save and load them.


```python title="Save and load configurations"
--8<-- "configuration_io.py:save_load"
```