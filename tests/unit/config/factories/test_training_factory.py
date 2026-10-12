from careamics.config.factories.training_factory import update_trainer_params


def test_update_trainer_params():
    """Test that the parameters are translated into the correct names."""
    num_epochs = 50
    num_steps = 1000
    train_params = {"check_val_every_n_epoch": 10}

    updated_dict = update_trainer_params(train_params)
    assert updated_dict == train_params

    updated_dict = update_trainer_params(
        train_params,
        num_epochs=num_epochs,
        num_steps=num_steps,
    )
    assert len(updated_dict) == 3
    assert updated_dict["max_epochs"] == num_epochs
    assert updated_dict["limit_train_batches"] == num_steps
