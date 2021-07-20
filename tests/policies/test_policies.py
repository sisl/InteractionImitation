import torch
from src.policies.policy import DeepSetsPolicy
import json5

config_path = "config/networks.json5"
with open(config_path, 'r') as cfg:
    config = json5.load(cfg)

def test_deepsets_policy():
    module = DeepSetsPolicy(config["ego_state"], config["deepsets"], config["path_encoder"], config["head"])
    
    ns = 5 
    nv = 7
    npath = 20

    ego_state = torch.rand(ns)
    relative_state = torch.rand(nv, ns)
    path_x = torch.rand(npath)
    path_y = torch.rand(npath)

    sample = {
        "state": ego_state,
        "relative_state": relative_state,
        "path_x": path_x,
        "path_y": path_y,
    }
    
    module(sample)