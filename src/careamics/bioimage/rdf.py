"""RDF related methods."""
from pathlib import Path


def _get_model_doc(name: str) -> str:
    """
    Return markdown documentation path for the provided model.

    Parameters
    ----------
    name : str
        Model's name.

    Returns
    -------
    str
        Path to the model's markdown documentation.

    Raises
    ------
    FileNotFoundError
        If the documentation file was not found.
    """
    doc = Path(__file__).parent.joinpath("docs").joinpath(f"{name}.md")
    if doc.exists():
        return str(doc.absolute())
    else:
        raise FileNotFoundError(f"Documentation for {name} was not found.")


def get_default_model_specs(
    name: str, mean: float, std: float, is_3D: bool = False
) -> dict:
    """
    Return the default bioimage.io specs for the provided model's name.

    Currently only supports `Noise2Void` model.

    Parameters
    ----------
    name : str
        Algorithm's name.
    mean : float
        Mean of the dataset.
    std : float
        Std of the dataset.
    is_3D : bool, optional
        Whether the model is 3D or not, by default False.

    Returns
    -------
    dict
        Model specs compatible with bioimage.io export.
    """
    rdf = {
        "name": "Noise2Void",
        "description": "Self-supervised denoising.",
        "license": "BSD-3-Clause",
        "authors": [
            {"name": "Alexander Krull"},
            {"name": "Tim-Oliver Buchholz"},
            {"name": "Florian Jug"},
        ],
        "cite": [
            {
                "doi": "10.48550/arXiv.1811.10980",
                "text": 'A. Krull, T.-O. Buchholz and F. Jug, "Noise2Void - Learning '
                'Denoising From Single Noisy Images," 2019 IEEE/CVF '
                "Conference on Computer Vision and Pattern Recognition "
                "(CVPR), 2019, pp. 2124-2132",
            }
        ],
        # "input_axes": ["bcyx"], <- overriden in save_as_bioimage
        "preprocessing": [  # for multiple inputs
            [  # multiple processes per input
                {
                    "kwargs": {
                        "axes": "zyx" if is_3D else "yx",
                        "mean": [mean],
                        "mode": "fixed",
                        "std": [std],
                    },
                    "name": "zero_mean_unit_variance",
                }
            ]
        ],
        # "output_axes": ["bcyx"], <- overriden in save_as_bioimage
        "postprocessing": [  # for multiple outputs
            [  # multiple processes per input
                {
                    "kwargs": {
                        "axes": "zyx" if is_3D else "yx",
                        "gain": [std],
                        "offset": [mean],
                    },
                    "name": "scale_linear",
                }
            ]
        ],
        "tags": ["unet", "denoising", "Noise2Void", "tensorflow", "napari"],
    }

    rdf["documentation"] = _get_model_doc(name)

    return rdf
