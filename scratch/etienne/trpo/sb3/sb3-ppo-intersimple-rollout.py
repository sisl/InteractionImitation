import sys
sys.path.append('..')

from stable_baselines3 import PPO
from core.sampling import rollout_sb3
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools
import torch
from wrappers import CollisionPenaltyWrapper

model = PPO.load('sb3-ppo-intersimple')
env = CollisionPenaltyWrapper(IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
), collision_distance=6, collision_penalty=100)

expert_data = rollout_sb3(env, model, n_episodes=200, max_steps_per_episode=200)

states, actions, rewards, dones = expert_data
print(f'Expert mean episode length {(~dones).sum() / states.shape[0]}')
print(f'Expert mean reward per episode {rewards[~dones].sum() / states.shape[0]}')

torch.save(expert_data, 'sb3-ppo-intersimple-expert-data.pt')
