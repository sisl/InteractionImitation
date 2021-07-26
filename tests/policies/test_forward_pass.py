import torch
import pickle
import json5
from src.bc import BehaviorCloningPolicy
config_path = "config/networks.json5"

with open(config_path, 'r') as cfg:
    config = json5.load(cfg)

filestr = 'tests/policies/base_'


def test_forward_pass():
    batch = pickle.load(open(filestr + '_test_batch.pkl', 'rb'))
    batch['relative_state'] = batch['relative_state'].float()
    policy = BehaviorCloningPolicy.load_model(config, filestr)
    policy.policy = policy.policy.type(batch['state'].dtype)
    action = policy(batch)
    i = torch.where(action.isnan())[0]
    for key in batch.keys():
        print(batch[key][i])
