from aenum import StrEnum

class SupportedActivation(StrEnum):
    """Supported activation functions.
    
    - None, no activation will be used.
    - Sigmoid
    - Softmax
    - Tanh
    - ReLU
    - LeakyReLU

    All activations are defined in PyTorch.

    See: https://pytorch.org/docs/stable/nn.html#loss-functions
    """
    _init_ = 'value __doc__'

    NONE = "None", "No activation will be used."
    SIGMOID = "Sigmoid", "Sigmoid activation function."
    SOFTMAX = "Softmax", "Softmax activation function."
    TANH = "Tanh", "Tanh activation function."
    RELU = "ReLU", "ReLU activation function."
    LEAKYRELU = "LeakyReLU", "LeakyReLU activation function."
