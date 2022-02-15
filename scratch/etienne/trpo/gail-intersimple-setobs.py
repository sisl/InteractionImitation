import gym
from core.gail import gail, Buffer
from core.value import SetValue
from core.policy import SetPolicy
from core.discriminator import DeepsetDiscriminator
import torch.optim
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools
from wrappers import CollisionPenaltyWrapper, Setobs
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

envs = [Setobs(TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
    stop_on_collision=False,
), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))) for _ in range(30)]
env_fn = lambda i: envs[i]

policy = SetPolicy(env_fn(0).action_space.shape[0])

value = SetValue()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-4)

discriminator = DeepsetDiscriminator()
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

expert_data = torch.load('intersimple-expert-data-setobs.pt')
expert_data = Buffer(*expert_data)

value, policy = gail(
    env_fn=env_fn,
    expert_data=expert_data,
    discriminator=discriminator,
    disc_opt=disc_opt,
    disc_iters=500,
    policy=policy,
    value=value,
    v_opt=v_opt,
    v_iters=1000,
    epochs=4000,
    rollout_episodes=30,
    rollout_steps=200,
    gamma=0.99,
    gae_lambda=0.9,
    delta=0.01,
    backtrack_coeff=0.8,
    backtrack_iters=10,
)

torch.save(policy.state_dict(), 'gail-intersimple-setobs.pt')
