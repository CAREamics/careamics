# Frequently asked questions

## Training

### `NotImplementedError: The operator [...] is not currently implemented for the MPS device`

Unfortunately, not all the operations in PyTorch are implemented for the Metal architecture 
of recent Apple devices (MPS stands for Metal Performance Shaders). This error occurs
when you are trying to run CAREamics on such a device without forcing training on CPU.

```python
NotImplementedError: The operator 'aten::max_pool3d_with_indices' is not currently implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
```

!!! success "Solution"
    In your environment (e.g. terminal before you start the jupyter notebook), set the
    environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1`. This will force the training
    to run on the CPU, which is slower but will work.

    This can be done for instance by running the following command in your terminal:

    ```bash
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    ```


## Prediction

### Sizes of tensors must match

If you get an error similar to:

```python
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 320 but got size 321 for tensor number 1 in the list.
```

This is likely because the input data size is not compatible with the model (e.g. the
model has layers that downsample the data, then upsample it, and this is only 
compatible with dimensions that are power of 2!).

!!! success "Solution"
    By [tiling](usage/prediction.md#tiling) the prediction, you can ensure that every
    input to the model has the correct shape. Then upon stitching back the tiles, you
    recover your whole image!


## BMZ model export

### `pydantic_core._pydantic_core.ValidationError`

The BMZ format is also validated by the [pydantic](https://pydantic-docs.helpmanual.io/)
library. If some of the metadata given to the `export_to_bmz` function is not correct, 
you might get such errors:

```python title="Code"
train_data = np.random.randint(0, 255, (256, 256)).
careamist.export_to_bmz( 
    path="n2v_models", 
    name="n2v_model_example.bmz", 
    input_array=train_data, 
    authors=[{ "name": "nobody", "email": "nobody@nobody", }]
)
```

```python title="Error message"
pydantic_core._pydantic_core.ValidationError: 2 validation errors for bioimage.io model specification
name
  Value error, 'n2v_model_example.bmz' is not restricted to 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_- ()' [type=value_error, input_value='n2v_model_example.bmz', input_type=str]
    For further information visit https://errors.pydantic.dev/2.7/v/value_error
authors.0.email
  value is not a valid email address: The part after the @-sign is not valid. It should have a period. [type=value_error, input_value='nobody@nobody', input_type=str]
```

If we look closely at the error message, Pydantic is telling us that there are actually
two errors:
- `name` should only contain letters, numbers, hyphens, underscores, spaces and parentheses.
- `authors.0.email` is not a valid email address, as it is missing a period after the @-sign.

!!! success "Solution"
    Look at the input `name`, it contains a forbidden character: the dot. The `email`
    should have something like `.com` after the `@` sign. Fixing these two issues will
    solve the problem.
