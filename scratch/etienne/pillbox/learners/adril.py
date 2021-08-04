import gym
from gym import spaces
from sklearn.neighbors import KDTree
from scipy.stats import norm
import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Generator, Optional, Union
import torch as th

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import ReplayBufferSamples, RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import ReplayBuffer


class AdRILWrapper(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, base_env):
    super(AdRILWrapper, self).__init__()
    self.base_env = base_env
    self.iter = 0
    self.observation_space = self.base_env.observation_space
    self.action_space = self.base_env.action_space
    self.trajs = list()
    self.num_trajs = 0
    self.curr_state = None
  def step(self, action):
    next_obs, _, done, info = self.base_env.step(action)
    reward = self.iter # Transformed by replay buffer
    self.trajs.append((self.curr_state, action, next_obs, done))
    if done:
        self.num_trajs += 1
    self.curr_state = next_obs
    return next_obs, reward, done, info
  def reset(self):
    obs = self.base_env.reset()
    self.curr_state = obs
    return obs
  def render(self, mode='human'):
    self.base_env.render(mode=mode)
  def close (self):
    self.base_env.close()
  def get_learner_trajs(self):
    return self.trajs
  def set_iter(self, k):
    self.iter = k
    
class AdRILReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        expert_data: dict = dict(),
            N_expert: int = 0,
        balanced: bool = True,
    ):
        super(AdRILReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, optimize_memory_usage=optimize_memory_usage)

        self.expert_states = expert_data['obs']
        self.expert_actions = expert_data['acts']
        self.expert_next_states = expert_data['next_obs']
        self.expert_dones = expert_data['dones']
        n_expert = len(expert_data["obs"])
        self.iter = 0
        self.N_expert = N_expert
        self.N_learner = 0
        self.normalizer = 1
        self.balanced = balanced

    def set_iter(self, k):
      self.iter = k
      normalizer = 0
      for i in range(0, k):
        normalizer += 1 ** (-i) # written to support decaying learning rate
      self.normalizer = normalizer

    def set_n_learner(self, n):
        self.N_learner = n

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        num_samples = len(batch_inds)
        if self.balanced:
          num_expert_samples = int(num_samples / 2)
          batch_inds = batch_inds[:num_expert_samples]
          expert_inds = np.random.randint(0, len(self.expert_states), size=num_expert_samples)
          # balanced sampling
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
          # AdRIL Rewards (indicator kernel)
          mask1 = (self.rewards[batch_inds] >= 0).astype(np.float32)
          mask2 = (self.rewards[batch_inds] < self.iter).astype(np.float32)
          r1 = - (1. ** (-self.rewards[batch_inds])) * mask1 * mask2 # Past iter
          r2 = np.zeros_like(self.rewards[batch_inds]) * mask1 * (1 - mask2) # current iter
          r3 = -self.rewards[batch_inds] * (1 - mask1) # Expert
          if self.iter > 0:
              rewards = (r1 / self.N_learner) + r2 + r3
          else:
              rewards = r1 + r2 + r3
          rewards = np.concatenate((rewards, np.ones_like(rewards) / self.N_expert), axis=0)
        else:
          if self.optimize_memory_usage:
              next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
          else:
              next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)
          obs = self._normalize_obs(self.observations[batch_inds, 0, :], env)
          actions = self.actions[batch_inds, 0, :]
          dones = self.dones[batch_inds]
          # AdRIL Rewards (indicator kernel)
          mask1 = (self.rewards[batch_inds] >= 0).astype(np.float32)
          mask2 = (self.rewards[batch_inds] < self.iter).astype(np.float32)
          r1 = - (1. ** (-self.rewards[batch_inds])) * mask1 * mask2 # Past iter
          r2 = np.zeros_like(self.rewards[batch_inds]) * mask1 * (1 - mask2) # current iter
          r3 = -self.rewards[batch_inds] * (1 - mask1) / self.N_expert # Expert
          if self.iter > 0:
              rewards = (r1 * 1. / self.N_learner) + r2 + r3
          else:
              rewards = r1 + r2 + r3
        data = (obs, actions, next_obs, dones, rewards)
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
