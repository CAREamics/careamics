#! /usr/bin/env python
"""Simple test used to check CAREamist import after packaging in the deployment CI."""


def test_smoke():
    """Import careamics.compat.CAREamist."""
    from careamics import CAREamistV2
    from careamics.config.ng_factories import create_n2v_config

    # create configuration
    config = create_n2v_config(
        experiment_name="smoke test",
        data_type="array",
        axes="YX",
        patch_size=(64, 64),
        batch_size=8,
    )
    CAREamistV2(source=config)


if __name__ == "__main__":
    test_smoke()
