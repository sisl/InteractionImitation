import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from typing import Callable, Union, Type, Optional, Dict, Any

# From https://github.com/DLR-RM/rl-baselines3-zoo/blob/8ea4f4a87afa548832ca17e575b351ec5928c1b0/utils/utils.py
def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

class SADataset(torch.utils.data.Dataset):
    def __init__(self, obs, acts, normalize):
        if normalize:
            obs = np.array(obs)
            self.mean = obs.mean(axis=0)
            self.std = obs.std(axis=0) + 1e-3
            obs = (obs - self.mean) / (self.std)
            self.is_normalized = True
        else:
            self.is_normalized = False
        self.obs = torch.tensor(obs)
        self.acts = torch.tensor(acts)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obs = self.obs[idx]
        acts = self.acts[idx]
        sample = {'obs': obs, 'acts': acts}
        return sample

def make_sa_dataloader(envname, max_trajs=None, normalize=False, batch_size=32):
    demos = np.load(
        "../experts/{0}/demos.npz".format(envname), allow_pickle=True)
    num_trajs = demos["num_trajs"]
    if max_trajs is None:
        max_trajs = num_trajs
    obs = []
    acts = []
    for traj in range(min(max_trajs, num_trajs)):
        obs.extend(demos[str(traj)].item()['states'])
        acts.extend(demos[str(traj)].item()['actions'])
    dataset = SADataset(obs, acts, normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)
    return dataloader

class SADSDataset(torch.utils.data.Dataset):
    def __init__(self, obs, acts, next_obs, traj_lens):
        self.obs = torch.tensor(obs)
        self.acts = torch.tensor(acts)
        self.next_obs = torch.tensor(next_obs)
        dones = [[False for _ in range(l - 2)] + [True] for l in traj_lens]
        self.dones = torch.tensor(list(chain.from_iterable(dones)))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        obs = self.obs[idx]
        acts = self.acts[idx]
        next_obs = self.next_obs[idx]
        dones = self.dones[idx]
        sample = {'obs': obs, 'acts': acts,
                  'next_obs': next_obs, 'dones': dones}
        return sample

def make_sads_dataloader(envname, max_trajs=None):
    demos = np.load(
        "./experts/{0}/demos.npz".format(envname), allow_pickle=True)
    num_trajs = demos["num_trajs"]
    if max_trajs is None:
        max_trajs = num_trajs
    obs = []
    next_obs = []
    acts = []
    lens = []
    for traj in range(min(max_trajs, num_trajs)):
        obs.extend(demos[str(traj)].item()['states'][:-1])
        next_obs.extend(demos[str(traj)].item()['states'][1:])
        acts.extend(demos[str(traj)].item()['actions'][:-1])
        lens.append(len(demos[str(traj)].item()['states']))
    dataset = SADSDataset(obs, acts, next_obs, lens)
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=False, num_workers=0, drop_last=True)
    return dataloader

def make_sa_dataset(envname, max_trajs=None):
    demos = np.load("../pillbox/experts/{0}/demos.npz".format(envname), allow_pickle=True)
    num_trajs = demos["num_trajs"]
    if max_trajs is None:
        max_trajs = num_trajs
    expert_states = []
    expert_actions = []
    expert_next_states = []
    expert_dones = []
    for traj in range(min(max_trajs, num_trajs)):
        expert_states.extend(demos[str(traj)].item()['states'][:-1])
        expert_next_states.extend(demos[str(traj)].item()['states'][1:])
        expert_actions.extend(demos[str(traj)].item()['actions'][:-1])
        l = len(demos[str(traj)].item()['states'])
        expert_dones.extend([False for _ in range(l - 2)] + [True])
    expert_data = dict()
    expert_data['obs'] = np.array(expert_states)
    expert_data['acts'] = np.array(expert_actions)
    expert_data['next_obs'] = np.array(expert_next_states)
    expert_data['dones'] = np.array(expert_dones)
    return expert_data
