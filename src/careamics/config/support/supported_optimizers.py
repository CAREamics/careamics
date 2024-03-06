from careamics.utils import BaseEnum


class SupportedOptimizer(str, BaseEnum):
    # ASGD = "ASGD"
    # Adadelta = "Adadelta"
    # Adagrad = "Adagrad"
    Adam = "Adam"
    # AdamW = "AdamW"
    # Adamax = "Adamax"
    # LBFGS = "LBFGS"
    # NAdam = "NAdam"
    # RAdam = "RAdam"
    # RMSprop = "RMSprop"
    # Rprop = "Rprop"
    SGD = "SGD"
    # SparseAdam = "SparseAdam"


class SupportedScheduler(str, BaseEnum):
    # ChainedScheduler = "ChainedScheduler"
    # ConstantLR = "ConstantLR"
    # CosineAnnealingLR = "CosineAnnealingLR"
    # CosineAnnealingWarmRestarts = "CosineAnnealingWarmRestarts"
    # CyclicLR = "CyclicLR"
    # ExponentialLR = "ExponentialLR"
    # LambdaLR = "LambdaLR"
    # LinearLR = "LinearLR"
    # MultiStepLR = "MultiStepLR"
    # MultiplicativeLR = "MultiplicativeLR"
    # OneCycleLR = "OneCycleLR"
    # PolynomialLR = "PolynomialLR"
    ReduceLROnPlateau = "ReduceLROnPlateau"
    # SequentialLR = "SequentialLR"
    StepLR = "StepLR"
