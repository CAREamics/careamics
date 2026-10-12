"""N2N references."""

from bioimageio.spec.model.v0_5 import CiteEntry, Doi

N2N_REF = CiteEntry(
    text="Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., "
    'Aittala, M. and Aila, T., 2018. "Noise2Noise: Learning image restoration '
    'without clean data". arXiv preprint arXiv:1803.04189.',
    doi=Doi("10.48550/arXiv.1803.04189"),
)
