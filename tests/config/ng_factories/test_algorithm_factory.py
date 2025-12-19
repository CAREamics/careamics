from careamics.config.ng_factories.algorithm_factory import (
    create_algorithm_configuration,
)


class TestUNetConfiguration:

    def test_model_creation(self):
        """Test that the correct parameters are passed to the model with priority to
        the explicit parameters."""
        model_kwargs = {
            "depth": 4,
            "conv_dims": 2,
            "n2v2": False,
            "in_channels": 2,
            "num_classes": 5,
            "independent_channels": False,
        }

        # choose different parameters
        conv_dims = 3
        in_channels = 3
        num_classes = 4
        independent_channels = True
        use_n2v2 = True

        config = create_algorithm_configuration(
            dimensions=conv_dims,
            algorithm="care",
            loss="mae",
            n_channels_in=in_channels,
            n_channels_out=num_classes,
            independent_channels=independent_channels,
            use_n2v2=use_n2v2,
            model_params=model_kwargs,
        )

        assert config["model"].depth == model_kwargs["depth"]
        assert config["model"].conv_dims == conv_dims
        assert config["model"].n2v2 == use_n2v2
        assert config["model"].in_channels == in_channels
        assert config["model"].num_classes == num_classes
        assert config["model"].independent_channels == independent_channels
