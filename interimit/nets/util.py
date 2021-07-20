import torch
from torch.nn import functional

def parse_functional(functional_config):
    if functional_config is None:
        return None
    elif isinstance(functional_config, str):
        if functional_config == 'relu':
            return functional.relu
        elif functional_config == 'sigmoid':
            return functional.sigmoid
        elif functional_config == 'softmax':
            return functional.softmax
    