import numpy as np

from careamics import CAREamist
from careamics.model_io.model_io_utils import export_bmz


def test_export_bmz(tmp_path, pre_trained):
    # training data
    train_array = np.ones((32, 32), dtype=np.float32)

    # instantiate CAREamist
    careamist = CAREamist(source=pre_trained, work_dir=tmp_path)

    # predict
    predicted = careamist.predict(train_array, tta_transforms=False)

    # save images
    train_path = tmp_path / "train.npy"
    np.save(train_path, train_array[np.newaxis, np.newaxis, ...])

    predicted_path = tmp_path / "predicted.npy"
    np.save(tmp_path / "predicted.npy", predicted[np.newaxis, ...])

    # export to BioImage Model Zoo
    export_bmz(
        model=careamist.model,
        config=careamist.cfg,
        path=tmp_path / "model.zip",
        name="TopModel",
        general_description="A model that just walked in.",
        authors=[{"name": "Amod", "affiliation": "El"}],
        inputs=train_path,
        outputs=predicted_path,
    )
