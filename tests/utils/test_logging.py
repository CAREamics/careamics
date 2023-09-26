from pathlib import Path

import pytest

from careamics.utils.logging import ProgressBar, get_logger


def test_get_logger(tmp_path: Path):
    logger = get_logger("test", log_path=tmp_path / "test.log")
    logger.info("test")
    assert (tmp_path / "test.log").exists()


@pytest.mark.parametrize(
    "max_value, epoch, num_epochs, mode",
    [(10, 0, 0, "predict"), (None, 0, 0, "predict")],
)
def test_progress_bar_update(max_value, epoch, num_epochs, mode):
    progress_bar = ProgressBar(
        max_value=max_value, epoch=epoch, num_epochs=num_epochs, mode=mode
    )
    for step in range(2):
        progress_bar.update(step, batch_size=1)
        assert progress_bar._seen_so_far == step

        if progress_bar.max_value is not None:
            assert progress_bar.spin is None
        else:
            assert next(progress_bar.spin)
