---
description: CAREamist API main page.
---

# CAREamist API

The CAREamist API is the recommended way to use CAREamics, it is a two stage process, in
which users first define a configuration and then use a the `CAREamist` to run their 
training and prediction.

## Quick start

The simplest way to use CAREamics is to create a configuration using the [convenience functions](configuration/convenience_functions.md). Checkout the [applications](../../applications/index.md) section for real-world examples of the various algorithms.

=== "Noise2Void"
    
    ```python
    --8<-- "careamist_api.py:quick_start_n2v"
    ```

    1. Obviously, choose a more realistic number of epochs for training.

    2. Use real data for training!


=== "CARE"

    ```python
    --8<-- "careamist_api.py:quick_start_care"
    ```

    1. Obviously, choose a more realistic number of epochs for training.

    2. Use real data for training! Here, we added validation data as well.


=== "Noise2Noise"

    ```python
    --8<-- "careamist_api.py:quick_start_n2n"
    ```

    1. Obviously, choose a more realistic number of epochs for training.

    2. Use real data for training!


## Documentation

There are many features that can be useful for your application, explore the
documentation to learn all the various aspects of CAREamics.

<div class="md-container secondary-section">
    <div class="g">
        <div class="section">
            <div class="component-wrapper" style="display: block;">
                <!-- New row -->
                <div class="responsive-grid">
                    <!-- Installation -->
                    <a class="card-wrapper" href="configuration">
                        <div class="card"> 
                            <div class="card-body"> 
                                <div class="logo">
                                    <span class="twemoji">
                                        --8<--  "tasklist.svg"
                                    </span>
                                </div>
                                <div class="card-content">
                                    <h5>Configuration</h5>
                                    <p>
                                        The configuration is at the heart of CAREamics, it 
                                        allow users to define how and which algorithm will be
                                        trained.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </a>
                    <!-- Installation -->
                    <a class="card-wrapper" href="usage">
                        <div class="card"> 
                            <div class="card-body"> 
                                <div class="logo">
                                    <span class="twemoji">
                                        --8<--  "code.svg"
                                    </span>
                                </div>
                                <div class="card-content">
                                    <h5>Using CAREAmics</h5>
                                    <p>
                                        The CAREamist is the core element allowing training
                                        and prediction using the model defined in the configuration.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>