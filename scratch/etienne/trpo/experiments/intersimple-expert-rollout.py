import torch
import functools
from core.sampling import rollout_sb3
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
from intersim.expert import NormalizedIntersimpleExpert
from util.wrappers import CollisionPenaltyWrapper

env = CollisionPenaltyWrapper(IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
), collision_distance=6, collision_penalty=100)
policy = NormalizedIntersimpleExpert(env.env, mu=0.001)

expert_data = rollout_sb3(env, policy, n_episodes=64, max_steps_per_episode=200)

states, actions, rewards, dones = expert_data
print(f'Expert mean episode length {(~dones).sum() / states.shape[0]}')
print(f'Expert mean reward per episode {rewards[~dones].sum() / states.shape[0]}')

torch.save(expert_data, 'intersimple-expert-data.pt')
