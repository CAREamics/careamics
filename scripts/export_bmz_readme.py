#!/usr/bin/env python
"""Export a README file for the bioimage model zoo."""
from pathlib import Path

from careamics.config import create_n2v_configuration
from careamics.model_io.bioimage._readme_factory import readme_factory


def main():
    # create configuration
    config = create_n2v_configuration(
        experiment_name="export_bmz_readme",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=2,
        num_epochs=10,
    )
    # export README
    readme_path = readme_factory(
        config=config, careamics_version="0.1.0", data_description="Mydata"
    )

    # copy file to __file__
    readme_path.rename(Path(__file__).parent / "README.md")


if __name__ == "__main__":
    main()
