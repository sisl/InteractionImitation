import torch
import gym
from stable_baselines3.common.vec_env import DummyVecEnv as VecEnv
from tqdm import tqdm
import numpy as np

def rollout(env_fn, policy, n_episodes, max_steps_per_episode):
    env = env_fn(0)
    states = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.observation_space.shape)
    actions = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.action_space.shape)
    rewards = torch.zeros(n_episodes, max_steps_per_episode + 1)
    dones = torch.ones(n_episodes, max_steps_per_episode + 1, dtype=bool)
    collisions = torch.zeros(n_episodes, max_steps_per_episode, dtype=bool)

    env = VecEnv(list(map(lambda i: (lambda: env_fn(i)), range(n_episodes))))

    states[:, 0] = torch.tensor(env.reset()).clone().detach()
    dones[:, 0] = False

    for s in tqdm(range(max_steps_per_episode), 'Rollout'):
        actions[:, s] = policy.sample(policy(states[:, s])).clone().detach()

        clipped_actions = actions[:, s]
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_actions = torch.clamp(clipped_actions, torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

        o, r, d, info = env.step(clipped_actions)
        states[:, s + 1] = torch.tensor(o).clone().detach()
        rewards[:, s] = torch.tensor(r).clone().detach()
        dones[:, s + 1] = torch.tensor(d).clone().detach()
        collisions[:, s] = torch.from_numpy(np.stack([
            i['collision'] for i in info
        ])).detach().clone()

    dones = dones.cumsum(1) > 0

    states = states[:, :max_steps_per_episode]
    actions = actions[:, :max_steps_per_episode]
    rewards = rewards[:, :max_steps_per_episode]
    dones = dones[:, :max_steps_per_episode]

    return states, actions, rewards, dones, collisions


def rollout_sb3(env, policy, n_episodes, max_steps_per_episode):
    states = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.observation_space.shape)
    actions = torch.zeros(n_episodes, max_steps_per_episode + 1, *env.action_space.shape)
    rewards = torch.zeros(n_episodes, max_steps_per_episode + 1)
    dones = torch.ones(n_episodes, max_steps_per_episode + 1, dtype=bool)

    for e in tqdm(range(n_episodes)):
        states[e, 0] = torch.tensor(env.reset()).clone().detach()
        dones[e, 0] = False

        for s in range(max_steps_per_episode):
            action, _ = policy.predict(states[e, s])
            actions[e, s] = torch.tensor(action).clone().detach()

            clipped_actions = actions[e, s]
            if isinstance(env.action_space, gym.spaces.Box):
                clipped_actions = torch.clamp(clipped_actions, torch.from_numpy(env.action_space.low), torch.from_numpy(env.action_space.high))

            o, r, d, _ = env.step(clipped_actions)
            states[e, s + 1] = torch.tensor(o).clone().detach()
            rewards[e, s] = torch.tensor(r).clone().detach()
            dones[e, s + 1] = torch.tensor(d).clone().detach()

            if d:
                break

    dones = dones.cumsum(1) > 0

    states = states[:, :max_steps_per_episode]
    actions = actions[:, :max_steps_per_episode]
    rewards = rewards[:, :max_steps_per_episode]
    dones = dones[:, :max_steps_per_episode]

    return states, actions, rewards, dones
