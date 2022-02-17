import gym
from core.sampling import rollout
from core.trpo import trpo
from core.value import Value
from core.policy import Policy
import torch.optim
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools
import numpy as np
from gym.wrappers import TransformObservation

obs_min = np.array([
    [-1000, -1000, 0, -np.pi, -1e-1, 0.],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
    [0, -np.pi, -20, -20, -np.pi, -1e-1],
]).reshape(-1)

obs_max = np.array([
    [1000, 1000, 20, np.pi, 1e-1, 0.],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
    [50, np.pi, 20, 20, np.pi, 1e-1],
]).reshape(-1)

envs = [TransformObservation(IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=10
    ),
), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10)) for _ in range(50)]

env_fn = lambda i: envs[i]
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-4, weight_decay=1e-3)

value, policy = trpo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=4000,
    rollout_episodes=30,
    rollout_steps=100,
    gamma=0.99,
    gae_lambda=0.95,
    delta=0.01,
    backtrack_coeff=0.9,
    backtrack_iters=50,
    v_opt=v_opt,
    v_iters=1000,
    cg_damping=0.1,
)

#rollout(env_fn, policy, n_episodes=9, max_steps_per_episode=200, render=True)
