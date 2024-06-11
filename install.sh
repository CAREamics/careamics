mamba install pytorch-gpu cuda-toolkit=11.8 torchvision -c nvidia -c conda-forge -c pytorch -y
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge tensorboard -y
mamba install -c anaconda scikit-learn -y
mamba install -c conda-forge matplotlib -y
mamba install -c anaconda ipython -y
mamba install -c conda-forge tifffile -y
mamba install -c anaconda ipykernel -y
mamba install -c conda-forge zarr -y
mamba install scikit-image -y
python -m pip install ml-collections 
python -m pip install pydantic
python -m pip install bioimageio.core
python -m pip install nbformat