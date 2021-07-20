import torch
import random
from src.nets import deepsets as ds
import copy
import numpy as np

ds_config = {
    "input_dim": 5,
    "phi": {
        "hidden_n": 1,
        "hidden_dim": 10,
    },
    "latent_dim": 8,
    "rho": {
        "hidden_n": 1,
        "hidden_dim": 10,
    },
    "output_dim" : 2,
}

def test_constructor():
    m = ds.DeepSetsModule.from_config(ds_config)
    phi_config = copy.deepcopy(ds_config["phi"])
    phi_config["input_dim"] = 5
    phi_config["output_dim"] = 2
    phi_config["final_activation"] = "sigmoid"
    phi = ds.Phi.from_config(phi_config)
    assert phi.final_activation == torch.sigmoid

    phi_config["final_activation"] = "relu"
    phi = ds.Phi.from_config(phi_config)
    assert phi.final_activation == torch.nn.functional.relu

def test_phi():
    input_dim = 5
    phi = ds.Phi(input_dim, 1, 10, 2)
    x = torch.rand(7, input_dim)
    y = phi(x)
    assert y.shape == torch.Size([7, 2])

    y = phi(torch.rand(input_dim))
    y = phi(torch.rand(7,7,7,input_dim))

def test_deepsets():
    m = ds.DeepSetsModule.from_config(ds_config)

    input_dim = ds_config["input_dim"]
    B = 50
    max_V = 10
    batch = []
    for i in range(B):
        if i==0: # ensure that there is an example with no vehicles
            n_dynamic = 0
        elif i==1: # and one with full vehicles
            n_dynamic = max_V
        else:
            n_dynamic = random.randint(1, max_V)
        x = torch.rand(n_dynamic, input_dim)
        n_nan = max_V - n_dynamic
        x = torch.cat([x, torch.zeros(n_nan, input_dim) * np.nan])
        assert x.shape == torch.Size([max_V, input_dim])
        batch.append(x)
    batch = torch.stack(batch)
    assert batch.shape == torch.Size([B, max_V, input_dim])

    y = m(batch)
    assert y.shape == torch.Size([B, ds_config["output_dim"]])
    assert torch.isnan(y).sum() == 0

    for i in range(B):
        y = m(batch[i])
        assert y.shape == torch.Size([ds_config["output_dim"]])
        assert torch.isnan(y).sum() == 0

def test_deepsets_computation():
    input_dim = 5
    latent_dim = 8

    B = 20
    max_V = 10
    batch = []
    for i in range(B):
        if i==0: # TODO: Set to i==0
            n_dynamic = 0
        else:
            n_dynamic = random.randint(1, max_V)
        x = torch.rand(n_dynamic, input_dim)
        n_nan = max_V - n_dynamic
        x = torch.cat([x, torch.zeros(n_nan, input_dim) * np.nan])
        assert x.shape == torch.Size([max_V, input_dim])
        batch.append(x)
    batch = torch.stack(batch)
    assert batch.shape == torch.Size([B, max_V, input_dim])
    x = batch

    ### create phi
    phi = ds.Phi(input_dim, 1, 10, latent_dim)

    max_nv = x.shape[-2]
    input_mask = ~torch.isnan(x)
    batch_dynamic_mask = torch.all(input_mask, dim=-1)
    assert batch_dynamic_mask.shape == x.shape[:-1]
    batch_mask = torch.all(batch_dynamic_mask, dim=-1)
    assert batch_mask.shape == x.shape[:-2]

    batch_dims = x.shape[:-2]
    latent = torch.zeros([*batch_dims, max_nv, latent_dim])
    latent[batch_dynamic_mask] = phi(x[batch_dynamic_mask])
    assert x[batch_dynamic_mask].shape == torch.Size([batch_dynamic_mask.sum(), input_dim])
    assert phi(x[batch_dynamic_mask]).shape == torch.Size([batch_dynamic_mask.sum(), latent_dim])

    latent = latent.sum(dim=-2)
    assert latent.shape == torch.Size([B, latent_dim])
