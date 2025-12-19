from careamics.config.ng_factories.algorithm_factory import _create_unet_configuration


class TestUNetConfiguration:

    def test_model_creation(self):
        """Test that the correct parameters are passed to the model."""
        model_kwargs = {
            "depth": 4,
            "conv_dims": 2,
            "n2v2": False,
            "in_channels": 2,
            "num_classes": 5,
            "independent_channels": False,
        }

        # choose different parameters
        axes = "XYZ"
        conv_dims = 3
        in_channels = 3
        num_classes = 4
        independent_channels = True
        use_n2v2 = True

        model = _create_unet_configuration(
            axes=axes,
            n_channels_in=in_channels,
            n_channels_out=num_classes,
            independent_channels=independent_channels,
            use_n2v2=use_n2v2,
            model_params=model_kwargs,
        )

        assert model.depth == model_kwargs["depth"]
        assert model.conv_dims == conv_dims
        assert model.n2v2 == use_n2v2
        assert model.in_channels == in_channels
        assert model.num_classes == num_classes
        assert model.independent_channels == independent_channels
