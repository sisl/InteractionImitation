import torch
import functools
from core.sampling import rollout_sb3
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
from intersim.expert import NormalizedIntersimpleExpert
from util.wrappers import CollisionPenaltyWrapper, Setobs
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

env = IntersimpleLidarFlatRandom(
    n_rays=5,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
)

policy = NormalizedIntersimpleExpert(env, mu=0.001)

env = Setobs(TransformObservation(
    CollisionPenaltyWrapper(
        env,
        collision_distance=6, collision_penalty=100
    ), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10)
))
expert_data = rollout_sb3(env, policy, n_episodes=2048, max_steps_per_episode=200)

states, actions, rewards, dones = expert_data
print(f'Expert mean episode length {(~dones).sum() / states.shape[0]}')
print(f'Expert mean reward per episode {rewards[~dones].sum() / states.shape[0]}')
print(f'Observation mean', states[~dones].mean(0))
print(f'Observation std', states[~dones].std(0))

torch.save(expert_data, 'intersimple-expert-data-setobs2.pt')
