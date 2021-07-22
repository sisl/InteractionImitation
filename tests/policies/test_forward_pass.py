import torch
import pickle
import json5
from src.bc import BehaviorCloningPolicy
config_path = "config/networks.json5"

with open(config_path, 'r') as cfg:
    config = json5.load(cfg)

filestr = 'tests/policies/base_'

batch = pickle.load(open(filestr+'_test_batch.pkl', 'rb'))
policy = BehaviorCloningPolicy.load_model(config, filestr)
action = policy(batch)