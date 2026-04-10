---
icon: lucide/file-question-mark
description: Frequently asked questions
---

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
