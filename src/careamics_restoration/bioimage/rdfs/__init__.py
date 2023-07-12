"""Provide supported models' default specs compatible with bioimage model zoo."""

from pathlib import Path
import yaml

from careamics_restoration.config.algorithm import Losses


def _load_model_rdf(name: str) -> dict:
    """Load an rdf(yaml) file given a model name."""
    name = name.lower()
    with open(Path(__file__).parent.joinpath(name + ".yaml")) as f:
        rdf = yaml.safe_load(f)

    # add covers if available
    cover_folder = Path(__file__).parent.parent.joinpath("covers")
    cover_images = list(cover_folder.glob(f"{name}_cover*.*"))
    if len(cover_images) > 0:
        rdf["covers"] = [str(img.absolute()) for img in cover_images]

    # add documentation (md file)
    doc_folder = Path(__file__).parent.parent.joinpath("docs")
    doc = list(doc_folder.glob(f"{name}.md"))
    if len(doc) > 0:
        rdf["documentation"] = doc[0]

    return rdf


# TODO: Keys are based on implemented losses, should it be more specific about models?
BIOIMAGE_MODEL_RDFS = {
    Losses.N2V: _load_model_rdf(Losses.N2V.value),
}


if __name__ == '__main__':
    print(BIOIMAGE_MODEL_RDFS["n2v"])
