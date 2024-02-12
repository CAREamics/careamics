from aenum import StrEnum

class SupportedActivation(StrEnum):

    NONE = "None"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    TANH = "Tanh"
    RELU = "ReLU"
    LEAKYRELU = "LeakyReLU"
