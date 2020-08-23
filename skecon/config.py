import torch

default_dtype = torch.get_default_dtype()
default_device = "cpu"

def set_default_dtype(dtype):
    default_dtype = dtype

def set_default_device(device):
    default_device = device

def get_default_dtype(dtype):
    return default_dtype

def get_default_device(device):
    return default_device
