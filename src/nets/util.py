import torch
from torch.nn import functional

def parse_functional(functional_config):
    if isinstance(functional_config, str):
        if functional_config == 'relu':
            return functional.relu
        elif functional_config == 'sigmoid':
            return torch.sigmoid
        elif functional_config == 'softmax':
            return functional.softmax
    return None