from tempfile import TemporaryDirectory

from careamics.config import create_n2v_config
from careamics.config.utils.configuration_io import load_configuration, save_configuration


# %%
# --8<-- [start:save_load]
config = create_n2v_config(
    experiment_name="n2v",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=1,
)

with TemporaryDirectory() as directory:
    path = save_configuration(config, f"{directory}/config.yml")
    loaded_config = load_configuration(path)
# --8<-- [end:save_load]
