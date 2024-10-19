"""
Contains various utility functions for PyTorch model building, training and saving.
"""

import torch
from torch import nn
import numpy as np
from pathlib import Path
import math
import random
import gc
from ptflops import get_model_complexity_info

def save_model(model: nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def number_of_patches(height: int, 
                      width: int, 
                      patch_size: int):
    """
    Returns the number of patches for a given image size and patch size.

    Args:
        height (int): The height of the input image.
        width (int): The width of the input image.
        patch_size (int): The size of each patch (assuming square patches).

    Returns:
        int: The number of patches that can be obtained from the input image.
    """
    return int((height * width) / patch_size**2)

def embedding_output_shape(height: int, 
                           width: int, 
                           patch_size: int, 
                           channels: int):
    """
    Calculates the output shape of an embedding layer based on input image dimensions, patch size, and number of channels.

    Args:
        height (int): The height of the input image.
        width (int): The width of the input image.
        patch_size (int): The size of each patch (assuming square patches).
        channels (int): The number of channels in the input image.

    Returns:
        tuple: A tuple representing the output shape (number of patches, patch size squared, number of channels).
    """
    return (number_of_patches(height, width, patch_size), patch_size**2, channels)

def is_torch_available():
    """
    Checks if the PyTorch library is available in the current environment.

    Returns:
        bool: True if PyTorch is available, False otherwise.
    """
    try:
        import torch
        return True
    except ImportError:
        return False

    
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

def count_parameters(model: nn.Module):
    """
    Calculate the total number of trainable parameters in a PyTorch model.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose parameters are to be counted.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def clear_model_from_memory(model): 
    """
    Clears the specified model from memory to free up resources.

    Args:
        model (torch.nn.Module): The PyTorch model to be cleared from memory.
    """
    del model
    gc.collect()
    torch.cuda.empty_cache()

def model_size_mb(model: nn.Module): 
    """
    Calculates the size of the model's parameters in megabytes (MB).

    Args:
        model (nn.Module): The PyTorch model whose parameter size is to be calculated.

    Returns:
        float: The size of the model's parameters in megabytes.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = total_params * 4
    size_in_mb = size_in_bytes / (1024 ** 2)
    return size_in_mb

def count_model_flops(model: nn.Module, input_size=(3, 224, 224), print_results=True):
    """
    Calculates and optionally prints the FLOPs (Floating Point Operations) and number of parameters 
    for a given PyTorch model.

    Args:
        model (nn.Module): The target PyTorch model for which to compute the FLOPs and parameters.
        input_size (tuple, optional): The size of the input tensor. Default is (3, 224, 224), 
                                       which represents a 3-channel image of size 224x224 pixels.
        print_results (bool, optional): If True, prints the calculated FLOPs and parameters. 
                                         Default is True.

    Returns:
        tuple: A tuple containing two elements:
            - gflops (float): The computed FLOPs of the model in GFLOPs (Giga Floating Point Operations).
            - params (float): The number of parameters in the model in millions (M).

    Notes:
        - The function uses the `ptflops` library to calculate the number of MACs (Multiply-Accumulate Operations) 
          and parameters. 
        - MACs are converted to GFLOPs by dividing by 1e9 (1 billion).
        - If an error occurs during computation, it will be printed, and the function will return (None, None).
    """
    try:
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False, verbose=False
        )

        gflops = macs / 1e9  # Convert MACs to GFLOPs

        if print_results:
            print(f'Computational complexity: {gflops:.3f} GFLOPs')
            print(f'Number of parameters: {params / 1e6:.3f} M')

        return gflops, params
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None