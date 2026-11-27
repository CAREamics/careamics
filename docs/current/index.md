---
icon: lucide/house
description: Quick start
---

# Using CAREamics

The CAREamist API is the recommended way to use CAREamics, it is generally a two stage
process in which users first define a configuration and then use a `CAREamist` instance
to run their training and prediction.

More advanced users wishing to have more control over the training and prediction
processes can re-use CAREamics modules in a [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable) script, which we refer
to as the [Lightning API](./lightning_api/lightning_api.md).

## Quick start

Check out the [configuration](./configuration) section for more details and parameters.

=== "Noise2Void"
    
    ```python title="Train Noise2Void"
    --8<-- "current/index/index_n2v.py:quick_start_n2v_config"
    --8<-- "current/index/index_n2v.py:quick_start_n2v_train"
    ```

    1. We will train from arrays, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. `train_data` must be compatible with `data_type`.


=== "CARE"

    ```python title="Train CARE"
    --8<-- "current/index/index_care.py:quick_start_care_config"
    --8<-- "current/index/index_care.py:quick_start_care_train"
    ```

    1. We will train from arrays, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. `train_data` must be compatible with `data_type`.

    3. Training CARE requires a target.



=== "HDN"

    ```python title="Train HDN"
    --8<-- "current/index/index_hdn.py:quick_start_hdn_config"
    --8<-- "current/index/index_hdn.py:quick_start_hdn_train"
    ```

    1. We will train from arrays, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. HDN may be trained using a [noise model](./noise_model).
    
    2. `train_data` must be compatible with `data_type`.

    4. Required if `use_noise_model=True`.




=== "MicroSplit"

    ```python title="Train MicroSplit"
    --8<-- "current/index/index_microsplit.py:quick_start_microsplit_config"
    --8<-- "current/index/index_microsplit.py:quick_start_microsplit_train"
    ```

    1. We will train from arrays, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. Number of channels produced by the channel-splitting model.

    3. Training MicroSplit requires a target.

    4. MicroSplit may require a [noise model](./noise_model).


=== "cryoCARE"

    ```python title="Train cryoCARE"
    --8<-- "current/index/index_cryocare.py:quick_start_cryocare_config"
    --8<-- "current/index/index_cryocare.py:quick_start_cryocare_train"
    ```

    1. We will train from `mrc` files, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. cryoCARE requires 3D stacks.

    3. Physical tilt axis used for reconstruction.

    4. Reconstruction from **even** angles is used as training input.

    5. Reconstruction from **odd** angles is used as training input.

    6. Reconstruction from **even** angles is used as prediction input.

    7. Reconstruction from **odd** angles is also used during prediction.


=== "Segmentation"

    ```python title="Train segmentation"
    --8<-- "current/index/index_segmentation.py:quick_start_segmentation_config"
    --8<-- "current/index/index_segmentation.py:quick_start_segmentation_train"
    ```

    1. We will train from arrays, but this could be `"tiff"` or `"zarr"` (see [data](./data/)).

    2. Number of foreground classes.

    3. Return probability map per class instead of a class mask.

    2. `train_data` must be compatible with `data_type`.

    5. Segmentation targets are integer label masks.



## Explore CAREamics


<div class="grid cards" markdown>


-   :lucide-sliders-horizontal:{ .lg .middle } __Configuration__

    ---

    Configure your experiment.

    [:octicons-arrow-right-24: Configuration](./configuration/index.md)

-   :lucide-database:{ .lg .middle } __Data preparation__

    ---

    Prepare your data.

    [:octicons-arrow-right-24: Data preparation](./data/data.md)


-   :lucide-network:{ .lg .middle } __Training__

    ---

    Train CAREamics.

    [:octicons-arrow-right-24: Training CAREamics](./careamist/careamist_training.md)


-   :lucide-chart-bell-curve:{ .lg .middle } __Noise model__

    ---

    Train a noise model for specific algorithms.

    [:octicons-arrow-right-24: Noise model](./noise_model/noise_model.md)

-   :lucide-chart-scatter:{ .lg .middle } __Logging__

    ---

    Follow the training progress and inspect the results.

    [:octicons-arrow-right-24: Logging](./careamist/careamist_train_logging.md)


-   :lucide-trending-up-down:{ .lg .middle } __Prediction__

    ---

    Use the trained model to predict.

    [:octicons-arrow-right-24: Prediction](./careamist/careamist_predicting.md)


-   :lucide-save:{ .lg .middle } __Saving models__

    ---

    Save checkpoints and export models to the BioImage Model Zoo.

    [:octicons-arrow-right-24: Saving and exporting models](./saving_models/saving_models.md)


-   :lucide-zap:{ .lg .middle } __Lightning API__

    ---

    Use CAREamics modules in your own PyTorch Lightning script.

    [:octicons-arrow-right-24: Lightning API](./lightning_api/lightning_api.md)

</div>
