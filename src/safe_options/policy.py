import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from src.core.policy import SetDiscretePolicy

class SetMaskedDiscretePolicy(SetDiscretePolicy):

    def forward(self, observation, safe_actions):
        return torch.cat((super().forward(observation), safe_actions), -1)
    
    def torch_dist(self, dist):
        logits = dist[..., :self.action_dim]
        z = dist[..., self.action_dim:]
        a = super().torch_dist(logits).probs
        return Categorical(probs=a*z)
    
    def unsafe_probability_mass(self, dist):
        logits = dist[..., :self.action_dim]
        z = dist[..., self.action_dim:]
        a = super().torch_dist(logits).probs
        return (a * (1 - z)).sum(-1)
    
    def predict(self, observations, state=None, episode_start=None, deterministic=True):
        observation = torch.tensor(observations['observation'])
        safe_actions = torch.tensor(observations['safe_actions'])
        if deterministic:
            _, actions = self.forward(observation, safe_actions).max(-1)
        else:
            actions = self.sample(self.forward(observation, safe_actions))
        return actions, None
    
    # def torch_dist_nomask(self, dist):
    #     print('no mask logprob')
    #     logits = dist[..., :self.action_dim]
    #     return super().torch_dist(logits)

    # def log_prob(self, dist, actions):
    #     return self.torch_dist_nomask(dist).log_prob(actions)
    
    # def kl_divergence(self, dist1, dist2):
    #     d1 = self.torch_dist_nomask(dist1)
    #     d2 = self.torch_dist_nomask(dist2)
    #     return kl_divergence(d1, d2)
