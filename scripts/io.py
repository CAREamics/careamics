from pathlib import Path
from typing import Literal


def list_files(
    datadir: str | Path,
    split: Literal["train", "val", "test"],
    subset: Literal["inputs", "targets"],
) -> list[Path]:
    files_dir = Path(datadir) / subset / split
    if not files_dir.is_dir():
        raise FileNotFoundError(files_dir)
    return sorted(p for p in files_dir.iterdir() if p.is_file())
