import torch
from src.policies.policy import DeepSetsPolicy
import json5

config_path = "config/networks.json5"
with open(config_path, 'r') as cfg:
    config = json5.load(cfg)

def test_deepsets_policy():
    module = DeepSetsPolicy(config)
    
    B = 25
    ns = 5 
    nv = 7
    npath = 20

    ego_state = torch.rand(B, ns)
    relative_state = torch.rand(B, nv, ns)
    path_x = torch.rand(B, npath)
    path_y = torch.rand(B, npath)

    sample = {
        "state": ego_state[0],
        "relative_state": relative_state[0],
        "path_x": path_x[0],
        "path_y": path_y[0],
    }
    module(sample)

    sample = {
        "state": ego_state,
        "relative_state": relative_state,
        "path_x": path_x,
        "path_y": path_y,
    }
    module(sample)
