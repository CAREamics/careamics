############################################
#   Utility Functions
############################################

import torch.optim as optim
import os
import torch
import torch.nn as nn
import sys
import torchvision
import numpy as np


def config_validator(cfg):
    '''
    Check config file for required parameters.
    '''
    required_params = ['experiment_name', 'model', 'loss_function', 'optimizer', 'scheduler', 'data_loader', 'trainer']
    #TODO validate train_single_epoch fucntion name, loss func name etc
    #assert hasattr ...
    # algorithm list, models list, loss functions list. All torch modules hasattr. 
    return cfg


def save_checkpoint(model, name, save_best):
    '''
    Save the model to a checkpoint file.
    '''
    if save_best:
        torch.save(model.state_dict(), 'best_checkpoint.pth')
    else:
        torch.save(model.state_dict(), 'checkpoint.pth')

def printNow(string,a="",b="",c="",d="",e="",f=""):
    print(string,a,b,c,d,e,f)
    sys.stdout.flush()


def imgToTensor(img):
    '''
    Convert a 2D single channel image to a pytorch tensor.
    '''
    img.shape=(img.shape[0],img.shape[1],1)
    imgOut = torchvision.transforms.functional.to_tensor(img.astype(np.float32))
    return imgOut


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std


def denormalize(x, mean, std):
    return x*std + mean


def getDevice():
    print("CUDA available?",torch.cuda.is_available())
    assert(torch.cuda.is_available())
    device = torch.device("cuda")
    return device


def add_noise(image, sigma):
    img = np.array(image).astype(np.float32)
    gauss = np.random.normal(0, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = img + gauss
    return np.clip(noisy, a_min=0, a_max=66000)


def add_poisson(image, scale=0, offset=0, adjust_background=0):
    """Add poisson noise to the image

    Parameters
    ----------
    image : numpy array
    scale : int
        Scaling value to control the magnitude. Default produces default destribution
    offset : int
        Offset for the pixel value range.
    adjust_background: int
        Add specified fixed value to the background

    Returns
    -------
    numpy array
        Input image with simulated poisson noise
    """
    image = image + adjust_background
    offset = image.min() if image.min() - offset < 0 else offset
    scale = 2.0 ** (0 - scale)
    scaled = (image.astype(np.float32) - offset) * scale
    return np.clip(np.random.poisson(scaled) / scale + offset, 0, 66000)


def export_model_to_zoo(model, path):
    """Export model to Bioimage model zoo

    Parameters
    ----------
    model : torch.nn.Module
        Model to be exported
    path : str
        Path to save the exported model
    """
    try:
        import bioimageio.core
    except ImportError:
        raise ImportError('bioimageio.core not found. Please install it first.')
    #TODO add model name, author, etc
    bioimageio.core.build_spec.build_model()


def export_model_to_onnx(model, path):
    """Export model to ONNX format

    Parameters
    ----------
    model : torch.nn.Module
        Model to be exported
    path : str
        Path to save the exported model
    """
    dummy_input = torch.randn(1, 1, 256, 256, device=getDevice())
    torch.onnx.export(model, dummy_input, path, verbose=True, input_names=['input'], output_names=['output'])