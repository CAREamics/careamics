---
icon: lucide/sliders-horizontal
description: Choose the CAREamics algorithm that matches your data and task
---

# Choosing an algorithm

CAREamics unifies several algorithm under a shared interface. The right algorithm for your needs depends mainly on the task:

- Denoising microscopy images:
    - Without ground truth: [Noise2Void and co](./n2v.md) (N2V, N2V2, structN2V), [HDN](./hdn.md)
    - With multiple noisy instances: Noise2Noise (see [CARE](./care.md))
    - With ground-truth: [CARE](./care.md)
- Denoising cryo-ET tomograms: [cryoCARE](./cryocare.md)
- Channel unmixing: [MicroSplit](./microsplit.md)
- Semantic segmentation: [Segmentation](./segmentation.md)



## Algorithm overview

After choosing an algorithm, continue to the corresponding page.


| Algorithm | Task | Target | Comments |
| --- | --- | --- | --- |
| [Noise2Void (N2V)](./n2v.md) | Image denoising | :lucide-circle-off: | Fast, self-supervised, limited by correlated noise and checkerboard artefacts |
| [N2V2](./n2v.md#n2v2) | Image denoising | :lucide-circle-off: | N2V variant designed to reduce checkerboard artefacts |
| [structN2V](./n2v.md#structn2v) | Image denoising | :lucide-circle-off: | N2V variant for structured noise |
| [HDN](./hdn.md) | Image denoising | :lucide-circle-off: | Variational self-supervised denoising using a [Noise model](./noise_model/noise_model.md) |
| [CARE](./care.md) | Image denoising | Paired clean-noisy images | Supervised denoising |
| [Noise2Noise (N2N)](./care.md#care-and-noisen2noise-differences) | Image denoising | Paired noisy image | Denoising using pairs of noisy images |
| [MicroSplit](./microsplit.md) | Channel unmixing | Individual channels | Supervised, may require a [noise model](./noise_model/noise_model.md) |
| [cryoCARE](./cryocare.md) | Cryo-ET volume denoising | Odd-even reconstructions | Requires pairs of odd and even reconstructions |
| [Segmentation](./segmentation.md) | Semantic segmentation | Label masks | Predicts class labels rather than denoised intensities |


