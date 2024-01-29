from enum import Enum

class SupportedActivation(str, Enum):

    NONE = "None"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    TANH = "Tanh"
    RELU = "ReLU"
    LEAKYRELU = "LeakyReLU"
