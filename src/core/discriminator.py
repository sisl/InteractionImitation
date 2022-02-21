import torch
import torch.nn as nn

class Discriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.nn = nn.Sequential(
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(1),
        )
    
    def forward(self, states, actions):
        return self.nn(torch.cat((states, actions), dim=-1)).squeeze(-1)

class DeepsetDiscriminator(nn.Module):

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
    
    def forward(self, states, actions):
        actions = actions.unsqueeze(-2)
        actions = actions.expand(*actions.shape[:-2], states.shape[-2], actions.shape[-1])
        sa = torch.cat((states, actions), dim=-1)
        return self.glob(self.elem(sa).sum(-2)).squeeze(-1)

class RecurrentDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.state_dim = 10
        self.state = nn.Sequential(
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(self.state_dim),
        )
        self.glob = nn.Sequential(
            nn.LazyLinear(10),
            nn.Tanh(),
            nn.LazyLinear(1),
        )
    
    def forward(self, states, actions):
        actions = actions.unsqueeze(-2)
        batch_size = actions.shape[:-2]
        set_size = states.shape[-2]
        action_dim = actions.shape[-1]
        actions = actions.expand(*batch_size, set_size, action_dim)
        sa = torch.cat((states, actions), dim=-1)

        state = torch.zeros((*batch_size, self.state_dim))
        for i in range(set_size):
            state = state + self.state(torch.cat((state, sa[..., i, :]), dim=-1))

        return self.glob(state).squeeze(-1)
