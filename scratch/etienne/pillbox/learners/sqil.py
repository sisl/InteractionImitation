import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer


class SQILReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        expert_data: dict = dict(),
    ):
        super(SQILReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage)

        self.expert_states = expert_data['obs']
        self.expert_actions = expert_data['acts']
        self.expert_next_states = expert_data['next_obs']
        self.expert_dones = expert_data['dones']

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        num_samples = len(batch_inds)
        num_expert_samples = int(num_samples / 2)
        batch_inds = batch_inds[:num_expert_samples]
        expert_inds = np.random.randint(0, len(self.expert_states), size=num_expert_samples)
        # Balanced sampling
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)
        next_obs = np.concatenate((next_obs, self._normalize_obs(self.expert_next_states[expert_inds], env)), axis=0)
        obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
        obs = np.concatenate((obs, self._normalize_obs(self.expert_states[expert_inds], env)), axis=0)
        actions = self.actions[batch_inds, 0, :]
        actions = np.concatenate((actions, self.expert_actions[expert_inds].reshape(num_expert_samples, -1)), axis=0)        
        dones = self.dones[batch_inds]
        dones = np.concatenate((dones, self.expert_dones[expert_inds].reshape(num_expert_samples, -1)), axis=0)
        # SQIL Rewards
        rewards = self.rewards[batch_inds] * 0.
        rewards = np.concatenate((rewards, np.ones_like(rewards)), axis=0)

        data = (obs, actions, next_obs, dones, rewards)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))