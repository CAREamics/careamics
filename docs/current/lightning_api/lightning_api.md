---
icon: lucide/zap
description: Lightning API
---

# Lightning API

CAREamics relies on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/?referrer=platform-docs), 
and thus advanced users can use the underlying modules in their own PyTorch Lightning scripts.
This is what we refer to as the Lightning API, and it is recommended for users who want
more control over the training and prediction process.

## Quick start

The Lightning API follows the same pattern for every algorithm: create a configuration,
instantiate the algorithm-specific Lightning module, create a `CareamicsDataModule`, and
pass both to a Lightning `Trainer`.

=== "Noise2Void"

    ```python
    --8<-- "current/lightning_api/lightning_api_n2v.py:lightning_api_n2v"
    --8<-- "current/lightning_api/lightning_api_n2v.py:lightning_api_n2v_2"
    ```

    1. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/n2v.md) section for more details.
    2. We instantiate the Lightning DataModule with the data configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input.
    For more details, refer to the [data documentation](../data/data.md).
    3. Each algorithm in CAREamics has its own Lightning module (the model), which 
    takes the algorithm part of the configuration.
    4. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    5. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    6. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    7. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    8. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    9. Create a new datamodule for inference, using the new data configuration.
    10. As for training, we predict using Lightning.
    11. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "CARE"

    ```python
    --8<-- "current/lightning_api/lightning_api_care.py:lightning_api_care"
    --8<-- "current/lightning_api/lightning_api_care.py:lightning_api_care_2"
    ```

    1. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/care.md) section for more details.
    2. The example creates synthetic input and target arrays, then instantiates the Lightning DataModule with the data configuration. It also takes `str`, `Path`, `numpy.ndarray`, or a list of those as input. For more details, refer to the [data documentation](../data/data.md).
    3. The `CAREModule` is the Lightning module for CARE and takes the algorithm part of the configuration.
    4. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    5. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    6. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    7. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    8. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    9. Create a new datamodule for inference, using the new data configuration.
    10. As for training, we predict using Lightning.
    11. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "Noise2Noise"

    ```python
    --8<-- "current/lightning_api/lightning_api_n2n.py:lightning_api_n2n"
    --8<-- "current/lightning_api/lightning_api_n2n.py:lightning_api_n2n_2"
    ```

    1. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/) section for more details.
    2. The example creates synthetic clean and noisy arrays, then instantiates the Lightning DataModule with the data configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input, and Noise2Noise uses separate noisy input and target arrays. For more details, refer to the [data documentation](../data/data.md).
    3. Noise2Noise uses the `CAREModule` Lightning module with its algorithm configuration.
    4. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    5. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    6. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    7. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    8. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    9. Create a new datamodule for inference, using the new data configuration.
    10. As for training, we predict using Lightning.
    11. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "HDN"

    ```python
    --8<-- "current/lightning_api/lightning_api_hdn.py:lightning_api_hdn"
    --8<-- "current/lightning_api/lightning_api_hdn.py:lightning_api_hdn_2"
    ```

    1. The example creates synthetic signal and observation arrays and trains a noise model from them. The generated noise-model files are used by HDN during training.
    2. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/) section for more details.
    3. The example instantiates the Lightning DataModule with the configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input; HDN trains on the observed noisy data. For more details, refer to the [data documentation](../data/data.md).
    4. The `HDNModule` is the Lightning module for HDN and takes the algorithm part of the configuration.
    5. Load one trained noise-model file for each model output channel with `set_noise_model`.
    6. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    7. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    8. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    9. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    10. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    11. Create a new datamodule for inference, using the new data configuration.
    12. As for training, we predict using Lightning.
    13. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "MicroSplit"

    ```python
    --8<-- "current/lightning_api/lightning_api_microsplit.py:lightning_api_microsplit"
    --8<-- "current/lightning_api/lightning_api_microsplit.py:lightning_api_microsplit_2"
    ```

    1. The example creates synthetic input, target, and noise-model training arrays, then trains and saves one noise model per output channel.
    2. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/) section for more details.
    3. The example instantiates the Lightning DataModule with the configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input, including separate target data for supervised training. For more details, refer to the [data documentation](../data/data.md).
    4. The `MicroSplitModule` is the Lightning module for MicroSplit and takes the algorithm part of the configuration.
    5. Load one trained noise-model file for each model output channel with `set_noise_model`.
    6. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    7. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    8. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    9. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    10. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    11. Create a new datamodule for inference, using the new data configuration.
    12. As for training, we predict using Lightning.
    13. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "cryoCARE"

    ```python
    --8<-- "current/lightning_api/lightning_api_cryocare.py:lightning_api_cryocare"
    --8<-- "current/lightning_api/lightning_api_cryocare.py:lightning_api_cryocare_2"
    ```

    1. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/) section for more details.
    2. The example creates synthetic input and target volumes, then instantiates the Lightning DataModule with the data configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input, including separate target volumes for supervised training. For more details, refer to the [data documentation](../data/data.md).
    3. The `CryoCAREModule` is the Lightning module for cryoCARE and takes the algorithm part of the configuration.
    4. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    5. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    6. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    7. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    8. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a 3D `new_patch_size` and `overlap_size` for tiled prediction.
    9. Create a new datamodule for inference, using the new data configuration.
    10. As for training, we predict using Lightning.
    11. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

=== "Segmentation"

    ```python
    --8<-- "current/lightning_api/lightning_api_segmentation.py:lightning_api_segmentation"
    --8<-- "current/lightning_api/lightning_api_segmentation.py:lightning_api_segmentation_2"
    ```

    1. Creating a CAREamics configuration ensures that the parameters passed
    to the various PyTorch modules are coherent. Check out the [configuration](../configuration/) section for more details.
    2. The example creates synthetic input images and integer target labels, then instantiates the Lightning DataModule with the data configuration. It takes `str`, `Path`, `numpy.ndarray`, or a list of those as input. For more details, refer to the [data documentation](../data/data.md).
    3. The `SegModule` is the Lightning module for segmentation and takes the algorithm part of the configuration.
    4. The CAREamics configuration has its own default set of parameters for the `ModelCheckpoint`
    callback (and `EarlyStopping`). You can either use those or set up your own.
    5. The `ConfigSaverCallback` callback stores the CAREamics configuration in the
    checkpoints. It is optional in the Lightning API, add it if you want to load the
    checkpoint with the CAREamist API and recover the training data and training configs. Without it, the checkpoint can still be loaded, but the training configuration falls back to its defaults. It is also required for checkpoints saved after `trainer.predict`: prediction replaces the trainer's datamodule, and only the callback keeps the training data configuration and its normalization statistics.
    6. The configuration creates the training-parameter configuration. You can set your own rather than reusing those.
    7. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
    8. Our data modules require a data configuration, we reuse part of the training data configuration, but convert it to a "predicting" mode. This gives the opportunity to change some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
    9. Create a new datamodule for inference, using the new data configuration.
    10. As for training, we predict using Lightning.
    11. Predictions are returned as a list of tiles because we used `new_patch_size` in the data configuration, therefore we need to stitch those tiles back together.

!!! warning "Missing validation"

    CAREamics performs additional validations, for instance using model constraints passed to the `CAREamicsDataModule`. When using the Lightning API, make sure that your data input and model have compatible shapes.


## Predicting directly to disk

A useful feature of CAREamics that can be leveraged in the Lightning API is writing
predictions directly to disk. This is achieved by adding a `PredictionWriterCallback`.

```python
--8<-- "current/lightning_api/lightning_api_n2v.py:predict_to_disk"
```

1. We keep the prediction writer in memory and disable writing, in case we want to
perform some prediction in memory first.
2. Add the prediction writer callback to the list of callbacks. This is a pre-requisite for writing to disk.
3. Once we are ready to predict to disk, we set a `writing_strategy` (`tiff`, `zarr`, or `custom`), and whether it is tiled.
4. We also need to turn back writing on.
5. Finally, we disable returning the predictions (unless you also want them returned in memory).
