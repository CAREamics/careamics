---
icon: lucide/git-branch
description: Choose the CAREamics algorithm that matches your data and task
---

# Choosing an algorithm

CAREamics folds several algorithm under a unified interface. The right algorithm for your needs depends mainly on what the task is:

- Denoising microscopy images: [Noise2Void (N2V)](./configuration/config_algorithm_specific/configuration_specific_n2v.md), [Noise2Noise](./configuration/config_algorithm_specific/configuration_specific_care.md), [HDN](./configuration/config_algorithm_specific/configuration_specific_hdn.md)
- Denoising cryo-ET tomograms: [cryoCARE](./configuration/config_algorithm_specific/configuration_specific_cryocare.md)
- Channel unmixing: [MicroSplit](./configuration/config_algorithm_specific/configuration_specific_microsplit.md)
- Semantic segmentation: [Segmentation](./configuration/config_algorithm_specific/configuration_seg.md)



## Algorithm overview

| Algorithm | Task | Target | Comments |
| --- | --- | --- | --- |
| [Noise2Void (N2V)](./configuration/config_algorithm_specific/configuration_specific_n2v.md) | Image denoising | None | fast, self-supervised, limited by correlated noise and checkerboard artefacts |
| [N2V2](./configuration/config_algorithm_specific/configuration_specific_n2v.md#n2v2) | Image denoising | None | N2V variant designed to reduce checkerboard artefacts |
| [structN2V](./configuration/config_algorithm_specific/configuration_specific_n2v.md#structn2v) | Image denoising | None | N2V variant for structured noise |
| [CARE](./configuration/config_algorithm_specific/configuration_specific_care.md) | Image denoising | Paired clean image | Supervised denoising from paired input and target images |
| [Noise2Noise (N2N)](./configuration/config_algorithm_specific/configuration_specific_care.md#care-and-noisen2noise-differences) | Image denoising | Paired noisy image | Uses an independent noisy acquisition as the target |
| [HDN](./configuration/config_algorithm_specific/configuration_specific_hdn.md) | Image denoising | Optional | supervised or self-supervised, may require a [Noise model](./noise_model/noise_model.md) |
| [MicroSplit](./configuration/config_algorithm_specific/configuration_specific_microsplit.md) | Channel unmixing | Individual channels | supervised, may require a [noise model](./noise_model/noise_model.md) |
| [cryoCARE](./configuration/config_algorithm_specific/configuration_specific_cryocare.md) | Cryo-ET volume denoising | Odd-even reconstructions | requires pairs of odd and even reconstructions |
| [Segmentation](./configuration/config_algorithm_specific/configuration_seg.md) | Semantic segmentation | Label masks | Predicts class labels rather than denoised intensities |


## Next steps

After choosing an algorithm, continue to the [Configuration guide](./configuration/configuration.md)
for shared data, patching, augmentation, normalization, and training parameters. Then
consult the algorithm-specific page for parameters and mechanisms that are
not shared.
