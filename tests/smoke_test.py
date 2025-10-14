#! /usr/bin/env python
"""Simple test used to check CAREamist import after packaging in the deployment CI."""


def test_smoke():
    """Import careamics.CAREamist."""
    from careamics import CAREamist
    from careamics.config import create_n2v_configuration

    # create configuration
    config = create_n2v_configuration(
        experiment_name="smoke test",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=8,
    )
    CAREamist(source=config)


if __name__ == "__main__":
    test_smoke()
