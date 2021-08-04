from typing import Any, Dict, List, Optional, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import BasePolicy, register_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork


class SoftQNetwork(QNetwork):
    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self.forward(observation)
        probs = nn.functional.softmax(q_values * 10, dim=1)
        m = th.distributions.Categorical(probs)
        action = m.sample().reshape(-1) 
        return action


class SQLPolicy(DQNPolicy):
    def make_q_net(self) -> SoftQNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None)
        return SoftQNetwork(**net_args).to(self.device)


SoftMlpPolicy = SQLPolicy

register_policy("SoftMlpPolicy", SoftMlpPolicy)
