from careamics.utils import BaseEnum


class SupportedCallback(str, BaseEnum):
    MODELCHECKPOINT = "ModelCheckpoint"
