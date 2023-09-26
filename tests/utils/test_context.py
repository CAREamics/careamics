from pathlib import Path

from careamics.utils import cwd


def test_cwd(tmp_path: Path):
    """Test cwd context manager."""
    current_path = Path(".").absolute()
    assert current_path != tmp_path.absolute()

    # change context
    with cwd(tmp_path):
        assert Path(".").absolute() == tmp_path.absolute()

        # create a file
        file_path = Path("test.txt")
        file_path.touch()
        assert Path(tmp_path, "test.txt").absolute().exists()

    # check that we are back to the original context
    assert Path(".").absolute() == current_path
