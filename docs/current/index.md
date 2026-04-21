---
icon: lucide/house
description: Quick start
---

# Using CAREamics

The CAREamist API is the recommended way to use CAREamics, it is a two stage process, in
which users first define a configuration and then use a the `CAREamist` to run their 
training and prediction.

More advanced users wishing to have more control over the training and prediction
process can re-use CAREamics module in a Pytorch Lightning script, which we refer
to as the [Lightning API](./lightning_api.md).

## Quick start

=== "Noise2Void"
    
    ```python
    --8<-- "current/getting_started.py:quick_start_n2v"
    ```

    1. This is training from arrays in memory, but this can also be done with files 
    on disk.

    2. The only important thing is that the data passed is coherent with the choice
    in the configuration.


=== "CARE"

    ```python
    --8<-- "current/getting_started.py:quick_start_care"
    ```

    1. This is training from arrays in memory, but this can also be done with files 
    on disk.

    2. The only important thing is that the data passed is coherent with the choice
    in the configuration.

## Explore CAREamics


<div class="grid cards" markdown>

-   :lucide-sliders-horizontal:{ .lg .middle } __Configuration__

    ---

    Configure your experiment.

    [:octicons-arrow-right-24: Configuration](./configuration.md)

-   :lucide-database:{ .lg .middle } __Data preparation__

    ---

    Prepare your data.

    [:octicons-arrow-right-24: Data preparation](./data.md)


-   :lucide-network:{ .lg .middle } __Training__

    ---

    Train CAREamics.

    [:octicons-arrow-right-24: Training CAREamics](./careamist_training.md)


-   :lucide-chart-scatter:{ .lg .middle } __Logging__

    ---

    Follow the training progress and inspect the results.

    [:octicons-arrow-right-24: Logging](./careamist_train_logging.md)


-   :lucide-trending-up-down:{ .lg .middle } __Prediction__

    ---

    Use the trained model to predict.

    [:octicons-arrow-right-24: Prediction](./careamist_predicting.md)


-   :lucide-zap:{ .lg .middle } __Lightning API__

    ---

    Use CAREamics modules in your own PyTorch Lightning script.

    [:octicons-arrow-right-24: Lightning API](./lightning_api.md)

</div>