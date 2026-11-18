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
    
    ```python
    --8<-- "current/index.py:quick_start_n2v"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. Data passed must be compatible with the configuration that was defined earlier.


=== "CARE"

    ```python
    --8<-- "current/index.py:quick_start_care"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. Data passed must be compatible with the configuration that was defined earlier.

    3. CARE requires a target.



=== "HDN"

    ```python
    --8<-- "current/index.py:quick_start_hdn"
    ```

    1. We declare that we will train from arrays in memory. See configuration section for more options.

    2. HDN may be trained using a [noise model](./noise_model).
    
    3. Data passed must be compatible with the configuration that was defined earlier.

    4. If a noise model was declared in the configuration, then it needs to supplied here.




=== "MicroSplit"

    ```python
    --8<-- "current/index.py:quick_start_microsplit"
    ```


=== "cryoCARE"

    ```python
    --8<-- "current/index.py:quick_start_cryocare"
    ```


=== "Segmentation"

    ```python
    --8<-- "current/index.py:quick_start_segmentation"
    ```



## Explore CAREamics


<div class="grid cards" markdown>

-   :lucide-git-branch:{ .lg .middle } __Choosing an algorithm__

    ---

    Select an algorithm based on your data, targets, and task.

    [:octicons-arrow-right-24: Choosing an algorithm](./choosing_an_algorithm.md)

-   :lucide-sliders-horizontal:{ .lg .middle } __Configuration__

    ---

    Configure your experiment.

    [:octicons-arrow-right-24: Configuration](./configuration.md)

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
