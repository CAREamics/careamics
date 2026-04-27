import pytest
import torch
from onnx import checker

from careamics.config.ng_factories import instantiate_algorithm_config
from careamics.lightning.dataset_ng.lightning_modules import get_module_cls


@pytest.mark.parametrize(
    "algorithm, architecture, conv_dim, n2v2, loss, shape",
    [
        ("n2n", "UNet", 2, False, "mae", (16, 16)),  # n2n 2D model
        ("n2n", "UNet", 3, False, "mae", (8, 16, 16)),  # n2n 3D model
        ("n2v", "UNet", 2, False, "n2v", (16, 16)),  # n2v 2D model
        ("n2v", "UNet", 3, False, "n2v", (8, 16, 16)),  # n2v 3D model
        ("n2v", "UNet", 2, True, "n2v", (16, 16)),  # n2v2 2D model
        ("n2v", "UNet", 3, True, "n2v", (8, 16, 16)),  # n2v2 3D model
    ],
)
def test_onnx_export(tmp_path, algorithm, architecture, conv_dim, n2v2, loss, shape):
    """Test model exportability to ONNX."""

    algo_config = {
        "algorithm": algorithm,
        "model": {
            "architecture": architecture,
            "conv_dims": conv_dim,
            "in_channels": 1,
            "num_classes": 1,
            "depth": 3,
            "n2v2": n2v2,
        },
        "loss": loss,
        "n2v_config": (
            None
            if algorithm != "n2v"
            else {"strategy": "median"} if n2v2 else {"strategy": "uniform"}
        ),
    }
    algo_config = instantiate_algorithm_config(algo_config)
    model = get_module_cls(algo_config.algorithm)(algo_config)
    model.model.eval()

    # create a sample input of BC(Z)XY
    x = torch.rand((1, 1, *shape))

    # create dynamic axes from the shape of the x
    dynamic_axes = {"input": {}, "output": {}}
    for i in range(len(x.shape)):
        dynamic_axes["input"][i] = f"dim_{i}"
        dynamic_axes["output"][i] = f"dim_{i}"

    torch.onnx.export(
        model,
        x,
        f"{tmp_path}/test_model.onnx",
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes=dynamic_axes,  # variable length axes,
    )

    checker.check_model(f"{tmp_path}/test_model.onnx")
