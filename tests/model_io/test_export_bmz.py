import numpy as np

from careamics import CAREamist
from careamics.model_io import export_to_bmz


def test_export_bmz(tmp_path, pre_trained):
    # training data
    train_array = np.ones((32, 32), dtype=np.float32)

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # predict (no tiling and no tta)
    predicted = careamist.predict(train_array, tta_transforms=False)

    # export to BioImage Model Zoo
    export_to_bmz(
        model=careamist.model,
        config=careamist.cfg,
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        input_array=train_array[np.newaxis, np.newaxis, ...],
        output_array=predicted,
    )
    assert (tmp_path / "model.zip").exists()
