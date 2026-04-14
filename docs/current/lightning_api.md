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

Here is an example of training Noise2Void using the Lightning API. For the configuration
parameters, refer to the [CAREamist API documentation]().

```python
--8<-- "current/lightning_api.py:lightning_api"
```

1. Creating a CAREamics configuration ensures that the parameters passed
to the various PyTorch modules are coherent.
2. `num_workers` is set here to "0" as it can often create issues on local machines (e.g.
Windows or macOS). Feel free to play with it!
3. Each class of algorithm in CAREamics has its own Lightning module (the model), here we
are setting up Noise2Void, but the same can be done for CARE and N2N using the `CAREModule`.
4. The Lightning Datamodule take `str`, `Path`, `numpy.ndarray` or list of those as input.
For more details, refer to the [data documentation]().
5. Careamics configuration has its own default set of parameters for the `ModelCheckpoint`
callback (and `EarlyStopping`). You can either use those or set up your own.
6. The `CareamicsCheckpointInfo` callback is used to log the configuration in the 
checkpoints.
7. Similarly, the configuration create training parameters configuration. You can set your
own rather than reusing those.
8. As in any Lightning script, pass the model and datamodule to the trainer and call `fit` to start training.
9. Our data modules require a data configuration, we reuse part of the training data 
configuration, but convert it to a "predicting" mode. This gives the opportunity to change
some parameters, such as passing a `new_patch_size` and `overlap_size` for tiled prediction.
10. Create a new datamodule for inference, using the new data configuration.
11. As for training, we predict using Lightning.
12. Predictions are returned as a list of tiles because we used `new_patch_size` in the
data configuration, therefore we need to stitch those tiles back together.

## Predicting directly to disk

A useful feature of CAREamics that can be leveraged in the Lightning API is writing
predictions directly to disk. This is achieved by adding a `PredictionWriterCallback`.

```python
--8<-- "current/lightning_api.py:lightning_api"
```

1. We keep the prediction writer in memory and disable writing, in case we want to
perform some prediction in memory first.
2. Add the prediction writer callback to the list of callbacks.
3. Once we are ready to predict to disk, we set a `writing_strategy` (`tiff`, `zarr`, or
`custom`), amd whether it is tiled.
4. We also need to turn back writing on.
5. Finally, we disable returning the predictions.