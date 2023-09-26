import pytest
import torch

from careamics_restoration.utils.torch_utils import (
    get_device,
    setup_cudnn_reproducibility,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_get_device(device):
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.gpu
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("benchmark", [True, False])
def test_setup_cudnn_reproducibility(deterministic, benchmark):
    setup_cudnn_reproducibility(deterministic=deterministic, benchmark=benchmark)
    assert torch.backends.cudnn.deterministic == deterministic
    assert torch.backends.cudnn.benchmark == benchmark
