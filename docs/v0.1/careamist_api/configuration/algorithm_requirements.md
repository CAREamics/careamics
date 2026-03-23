# Algorithm requirements

In this section we detail the constraints of each algorithm on the configuration and
their differences. 


### Parent configuration

The parent configuration is `Configuration` and it is the same for all algorithms. It 
should not be used to train any of the algorithms as the child classes ensure coherence
across the parameters.

### Noise2Void family

Noise2Void algorithms (N2V, N2V2, structN2V) are configured using `N2VConfiguration`.
It enforces the following constraints:


- `algorithm_config`: must be a `N2VAlgorithm`
- `algorithm_config.algorithm="n2v"`
- `algorithm_config.loss="n2v"`
- `algorithm_config.model`: 
    - must be a UNet (`architecture="UNet"`)
    - `in_channels` and `num_classes` must be equal
- `data_config`: must be a `N2VDataConfiguration`
- `data_config.transforms`: must contain `N2VManipulateModel` as the last transform


### CARE and Noise2Noise

The two algorithms are very similar and therefore their constraints are sensibly
the same. They are configured using `CAREConfiguration` and `N2NConfiguration`
respectively.

- `algorithm_config`: must be a `CAREAlgorithm` (CARE) or `N2NAlgorithm` (Noise2Noise)
- `algorithm_config.algorithm`: `care` (CARE) or `n2n` (Noise2Noise)
- `algorithm_config.loss`: `mae` or `mse`
- `algorithm_config.model`: must be a UNet (`architecture="UNet"`)
- `data_config`: must be a `DataConfiguration`
- `data_config.transforms`: must not contain `N2VManipulateModel`
