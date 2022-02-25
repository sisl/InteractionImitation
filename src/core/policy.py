import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, Categorical
from torch.distributions.kl import kl_divergence

class BasePolicy(nn.Module):
    
    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim

    def torch_dist(self, dist):
        return Independent(Normal(dist[..., :self.action_dim], dist[..., self.action_dim:].exp()), 1)
    
    def sample(self, dist):
        return self.torch_dist(dist).sample()
    
    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        observations = torch.tensor(observations)
        return self._predict(self.forward(observations), state, episode_start, deterministic)
    
    def _predict(self, dist, state=None, episode_start=None, deterministic=True):
        if deterministic:
            actions = dist[..., :self.action_dim]
        else:
            actions = self.sample(dist)
        return actions, None

    def log_prob(self, dist, actions):
        return self.torch_dist(dist).log_prob(actions)
    
    def kl_divergence(self, dist1, dist2):
        d1 = self.torch_dist(dist1)
        d2 = self.torch_dist(dist2)
        return kl_divergence(d1, d2)

class Policy(BasePolicy):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn.Sequential(
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(50),
            nn.Tanh(),
            nn.LazyLinear(2 * self.action_dim),
        )

    def forward(self, states):
        return self.nn(states)

class DiscretePolicy(BasePolicy):

    def __init__(self, *args, hidden_layer_size=50, n_hidden_layers=2, activation=nn.Tanh, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [nn.LazyLinear(hidden_layer_size), activation()] * n_hidden_layers
        self.nn = nn.Sequential(*layers, nn.LazyLinear(self.action_dim))

        #self.nn = nn.Sequential(
        #    nn.LazyLinear(hidden_layer_size),
        #    nn.Tanh(),
        #    nn.LazyLinear(hidden_layer_size),
        #    nn.Tanh(),
        #    nn.LazyLinear(self.action_dim),
        #)

    def forward(self, states):
        return self.nn(states)
    
    def torch_dist(self, dist):
        return Categorical(logits=dist)
    
    def _predict(self, dist, state=None, episode_start=None, deterministic=True):
        if deterministic:
            _, actions = dist.max(-1)
        else:
            actions = self.sample(dist)
        return actions, None

class SetPolicy(Policy):

    def forward(self, states):
        batch_size = states.shape[:-2]
        states = torch.cat((states[..., :1, [0, 1]], states[..., :, [2, 5]]), axis=-2).reshape(*batch_size, -1)
        return super().forward(states)

class SetDiscretePolicy(DiscretePolicy):

    def forward(self, states):
        batch_size = states.shape[:-2]
        states = torch.cat((states[..., :1, [0, 1]], states[..., :, [2, 5]]), axis=-2).reshape(*batch_size, -1)
        return super().forward(states)

class DeepSetPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            nn.LazyLinear(2 * self.action_dim),
        )
    
    def forward(self, states):
        return self.glob(self.elem(states).sum(-2))
