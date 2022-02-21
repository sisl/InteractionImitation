import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

class Value(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(1),
        )
    
    def forward(self, states):
        return self.nn(states).squeeze(-1)

class SetValue(Value):

    def forward(self, states):
        batch_size = states.shape[:-2]
        states = torch.cat((states[..., :1, [0, 1]], states[..., :, [2, 5]]), axis=-2).reshape(*batch_size, -1)
        return super().forward(states)

class DeepSetValue(nn.Module):

    def __init__(self):
        super().__init__()
        self.elem = nn.Sequential(
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(10),
        )
        self.glob = nn.Sequential(
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(1),
        )
    
    def forward(self, states):
        return self.glob(self.elem(states).sum(-2)).squeeze(-1)
