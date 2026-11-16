import numpy as np
import tifffile

from careamics.cli.main import main


def test_predict(tmp_path, pre_trained):
    image_path = tmp_path / "image.tiff"
    tifffile.imwrite(image_path, np.arange(32 * 32).reshape(32, 32))

    main(
        [
            "predict",
            "--ckpt_path",
            str(pre_trained),
            "--data",
            str(image_path),
            "--output_path",
            str(tmp_path),
            "--data_type",
            "tiff",
        ]
    )

    assert (tmp_path / "predictions" / "image.tiff").is_file()
