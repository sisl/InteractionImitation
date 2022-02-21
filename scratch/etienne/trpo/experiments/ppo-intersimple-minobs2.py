from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools

from core.ppo import ppo
from core.value import Value
from core.policy import Policy
import torch.optim
import numpy as np
from gym.wrappers import TransformObservation

from util.wrappers import Minobs

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

envs = [Minobs(TransformObservation(IntersimpleLidarFlatRandom(
    n_rays=5,
    reward=functools.partial(
        speed_reward,
        collision_penalty=1000
    ),
), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))) for _ in range(30)]

env_fn = lambda i: envs[i]
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3, weight_decay=1e-3)

value, policy = ppo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=50,
    rollout_episodes=30,
    rollout_steps=100,
    gamma=0.99,
    gae_lambda=0.9,
    clip_ratio=0.2,
    pi_opt=pi_opt,
    pi_iters=100,
    v_opt=v_opt,
    v_iters=1000,
)

torch.save(policy.state_dict(), 'ppo-intersimple.pt')
