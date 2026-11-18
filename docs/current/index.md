---
icon: lucide/house
description: Quick start
---

# Using CAREamics

The CAREamist API is the recommended way to use CAREamics, it is generally a two stage
process in which users first define a configuration and then use a `CAREamist` instance
to run their training and prediction.

More advanced users wishing to have more control over the training and prediction
process can re-use CAREamics module in a [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable) script, which we refer
to as the [Lightning API](./lightning_api/lightning_api.md).

## Quick start

=== "Noise2Void"
    
    ```python title="Create a configuration"
    --8<-- "current/index/index_n2v.py:quick_start_n2v_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_n2v.py:quick_start_n2v_train"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. Data passed must be compatible with the configuration that was defined earlier.


=== "CARE"

    ```python title="Create a configuration"
    --8<-- "current/index/index_care.py:quick_start_care_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_care.py:quick_start_care_train"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. Data passed must be compatible with the configuration that was defined earlier.

    3. CARE requires a target.



=== "HDN"

    ```python title="Create a configuration"
    --8<-- "current/index/index_hdn.py:quick_start_hdn_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_hdn.py:quick_start_hdn_train"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. HDN may be trained using a [noise model](./noise_model).
    
    3. Data passed must be compatible with the configuration that was defined earlier.

    4. If a noise model was declared in the configuration, then it needs to supplied here.




=== "MicroSplit"

    ```python title="Create a configuration"
    --8<-- "current/index/index_microsplit.py:quick_start_microsplit_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_microsplit.py:quick_start_microsplit_train"
    ```

    1. We declare that we will train from arrays in memory. See the configuration section for more options.

    2. `output_channels` is the number of channels produced by the channel-splitting model.

    3. `use_noise_model=True` enables the noise-model likelihood.

    4. A separate noise model is passed for each output channel, so the list contains three models.

    5. Data passed must be compatible with the configuration that was defined earlier.


=== "cryoCARE"

    ```python title="Create a configuration"
    --8<-- "current/index/index_cryocare.py:quick_start_cryocare_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_cryocare.py:quick_start_cryocare_train"
    ```

    1. `data_type="mrc"` selects cryoCARE's MRC input handling.

    2. cryoCARE uses volumes with three spatial axes (`ZYX`) and a physical tilt axis.

    3. `tilt_axis` identifies the physical tilt axis used to construct the split.

    4. `even_reconstruction` is passed as the training input.

    5. `odd_reconstruction` is passed as the training target.

    6. Both reconstructions are passed during prediction, and predictions are written to disk with `predict_to_disk`.

    7. Data passed must be compatible with the configuration that was defined earlier.


=== "Segmentation"

    ```python title="Create a configuration"
    --8<-- "current/index/index_segmentation.py:quick_start_segmentation_config"
    ```

    ```python title="Train and predict"
    --8<-- "current/index/index_segmentation.py:quick_start_segmentation_train"
    ```

    1. We declare that we will train from arrays in memory. See the configuration section for more options.

    2. Segmentation targets are integer label masks with one value per class.

    3. `n_classes` is the number of foreground classes; CAREamics adds the background class automatically.

    4. `prediction_output="probabilities"` returns one probability map per class instead of a class mask.

    5. Data passed must be compatible with the configuration that was defined earlier.



## Explore CAREamics


<div class="grid cards" markdown>

-   :lucide-git-branch:{ .lg .middle } __Choosing an algorithm__

    ---

    Select an algorithm based on your data, targets, and task.

    [:octicons-arrow-right-24: Choosing an algorithm](./choosing_an_algorithm.md)

-   :lucide-sliders-horizontal:{ .lg .middle } __Configuration__

    ---

    Configure your experiment.

    [:octicons-arrow-right-24: Configuration](./configuration/configuration.md)

-   :lucide-database:{ .lg .middle } __Data preparation__

    ---

    Prepare your data.

    [:octicons-arrow-right-24: Data preparation](./data/data.md)


-   :lucide-network:{ .lg .middle } __Training__

    ---

    Train CAREamics.

    [:octicons-arrow-right-24: Training CAREamics](./careamist/careamist_training.md)


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
