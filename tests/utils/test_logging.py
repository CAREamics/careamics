from pathlib import Path

from careamics.utils.logging import get_logger


def test_get_logger(tmp_path: Path):
    logger = get_logger("test", log_path=tmp_path / "test.log")
    logger.info("test")
    assert (tmp_path / "test.log").exists()
