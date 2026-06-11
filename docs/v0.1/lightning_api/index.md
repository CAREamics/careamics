---
description: Lightning API main page.
---

# Lightning API

!!! warning "Legacy documentation"
    This documentation is for the legacy version of CAREamics (v0.1), which is
    accessible through the `careamics.compat` module. It is kept here for reference, but
    we recommend using the latest version of CAREamics (v0.2) for new projects. Head to the [v0.2 guides](../v0.2/index.md).

The so-called "Lightning API" is how we refer to using the lightning modules 
from CAREamics in a [PyTorch Ligthning](https://lightning.ai/docs/pytorch/stable/) 
pipeline. In our [high-level API](../careamist_api/index.md), these modules are 
hidden from users and many checks, validations, error handling, and other 
features are provided. However, if you want to have increased flexibility, for instance
to use your own dataset, model or a different training loop, you can re-use many of 
CAREamics modules in your own PyTorch Lightning pipeline.


```python title="Basic Usage"
--8<-- "v0.1/lightning_api/basic_usage.py:basic_usage"
```

1. We provide convenience functions to create the various Lightning modules.

2. Each convenience function will have a set of algorithms. Often, these correspond 
to the parameters in the CAREamics configuration. You can check the next pages for more
details.

3. As for any Lightning pipeline, you need to instantiate a `Trainer`.

4. This way, you have all freedom to set your own callbacks.

5. Our prediction Lightning data module has the possibility to break the images into
overlapping tiles.

6. If you predicted using tiled images, you need to recombine the tiles into images.
We provide a general function to take care of this.


There are three types of Lightning modules in CAREamics:

- Lightning Module
- Training Lightning Datamodule
- Prediction Lightning Datamodule

In the next pages, we give more details on the various parameters of the convenience 
functions. For the rest, refer the the [PyTorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/).