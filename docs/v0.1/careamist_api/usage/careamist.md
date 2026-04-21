# CAREamist

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

The `CAREamist` is the central class in CAREamics, it provides the API to train, predict
and save models. There are three ways to create a `CAREamist` object: with a configuration, 
with a path to a configuration, or with a path to a trained model.


## Instantiating with a configuration

When passing a configuration to the `CAREamist` constructor, the model is initialized
with random weights and prediction will not be possible until the model is trained.


```python title="Instantiating CAREamist with a configuration"
--8<-- "v0.1/careamist_api/usage/careamist.py:config"
```

1. Any valid configuration will do!

!!! info "Creating configurations"
    Check out the [configuration guide](../configuration/index.md) to learn how to create
    configurations.


## Instantiating with a path to a configuration

This is similar to the previous section, except that the configuration is loaded from
a file on disk.

```python title="Instantiating CAREamist with a path to a configuration"
--8<-- "v0.1/careamist_api/usage/careamist.py:config_path"
```

## Instantiating with a path to a model

There are two types of models exported from CAREamics. During training, the model is
saved as checkpoints (`.ckpt`). After training, users can export the model to the 
bioimage model zoo format (saved as a`.zip`). Both can be loaded into CAREamics to
either retrain or predict. Alternatively, a checkpoint can be loaded in order to 
export it as a bioimage model zoo model.

In any case, both types of pre-trained models can be loaded into CAREamics by passing
the path to the model file. The instantiated CAREamist is then ready to predict on new
images!


```python title="Instantiating CAREamist with a path to a model"
--8<-- "v0.1/careamist_api/usage/careamist.py:load_model"
```

1. Any valid path to a model, as a string or a `Path.path` object, will work.


## Setting the working directory

By default, CAREamics will save the checkpoints in the current working directory. When
creating a new CAREamist, you can indicate a different working directory in which to
save the logs and checkpoints during training.

```python title="Changing the working directory"
--8<-- "v0.1/careamist_api/usage/careamist.py:work_dir"
```

## Custom callbacks

CAREamics uses two different callbacks from PyTorch Lightning:

- [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint): to save the model at different points during the training.
- [`EarlyStopping`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping): to stop the training based on a few parameters.

The parameters for the callbacks are the same as the ones from PyTorch Lightning, and
can be set in the configuration.

Custom callbacks can be passed to the `CAREamist` constructor. The callbacks must inherit
from the PyTorch Lightning `Callback` class.

```python title="Custom callbacks"
--8<-- "v0.1/careamist_api/usage/careamist.py:callbacks"
```

1. The callbacks must inherit from the PyTorch Lightning `Callback` class.

2. This is just an example to test that the callback was called!

3. Create your callback.

4. Pass the callback to the CAREamist constructor as a list.