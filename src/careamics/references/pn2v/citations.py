"""PN2V references."""

from bioimageio.spec.model.v0_5 import CiteEntry, Doi

PN2V_REF = CiteEntry(
    text="Krull, A., Vicar, T., Prakash, M., Lalit, M. and Jug, F., 2020. "
    '"Probabilistic noise2void: Unsupervised content-aware denoising". '
    "Frontiers in Computer Science, 2, p.5.",
    doi=Doi("10.3389/fcomp.2020.00005"),
)

N2V2_REF = CiteEntry(
    text="Höck, E., Buchholz, T.O., Brachmann, A., Jug, F. and Freytag, A., "
    '2022. "N2V2 - Fixing Noise2Void checkerboard artifacts with modified '
    'sampling strategies and a tweaked network architecture". In European '
    "Conference on Computer Vision (pp. 503-518).",
    doi=Doi("10.1007/978-3-031-25069-9_33"),
)

STRUCTN2V_REF = CiteEntry(
    text="Broaddus, C., Krull, A., Weigert, M., Schmidt, U. and Myers, G., 2020."
    '"Removing structured noise with self-supervised blind-spot '
    'networks". In 2020 IEEE 17th International Symposium on Biomedical '
    "Imaging (ISBI) (pp. 159-163).",
    doi=Doi("10.1109/isbi45749.2020.9098336"),
)
