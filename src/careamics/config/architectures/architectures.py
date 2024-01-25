from enum import Enum

class Architecture(str, Enum):
    
    UNET = "UNet"
    VAE = "VAE"


class Activation(str, Enum):

    NONE = "None"
    SIGMOID = "Sigmoid"
    SOFTMAX = "Softmax"
    TANH = "Tanh"
    RELU = "ReLU"
    LEAKYRELU = "LeakyReLU"
