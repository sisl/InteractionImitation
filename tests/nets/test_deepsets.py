import torch
import random
from src.nets import deepsets as ds
import copy

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
    n_dynamic = random.randint(5, 15)
    x = torch.rand(n_dynamic, input_dim)

    n_batch = 20
    x = x.unsqueeze(0).expand(n_batch, n_dynamic, input_dim)

    y = m(x)
    assert y.shape == torch.Size([n_batch, ds_config["output_dim"]])

    for i in range(n_batch):
        assert torch.allclose(y[i], y[0])

def test_deepsets_computation():
    n_dynamic = random.randint(5,15)
    n_batch = 7
    input_dim = 5
    output_dim = 3
    x = torch.rand(n_dynamic, input_dim)
    x = x.unsqueeze(0).expand(n_batch, n_dynamic, input_dim)
    assert x.shape == torch.Size([n_batch, n_dynamic, input_dim])

    phi = torch.nn.Linear(input_dim, output_dim)

    y = torch.stack(tuple(phi(instance) for instance in x.unbind(-2)), dim=-2)
    assert y.shape == torch.Size([n_batch, n_dynamic, output_dim])

    y = y.sum(dim=-2)
    assert y.shape == torch.Size([n_batch, output_dim])

    for i in range(n_batch):
        assert torch.allclose(y[i], y[0])