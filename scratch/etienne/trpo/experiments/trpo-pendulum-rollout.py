import gym
from core.gail import gail
from core.reparam_module import ReparamPolicy
from core.value import Value
from core.policy import Policy
from core.discriminator import Discriminator
import torch.optim
from core.sampling import rollout

env_fn = lambda _: gym.make('Pendulum-v0')
policy = Policy(env_fn(0).action_space.shape[0])
policy(torch.zeros(env_fn(0).observation_space.shape))
policy = ReparamPolicy(policy)
policy.load_state_dict(torch.load('trpo-pendulum.pt'))

expert_data = rollout(env_fn, policy, n_episodes=20, max_steps_per_episode=200)
torch.save(expert_data, 'trpo-pendulum-expert-data.pt')
