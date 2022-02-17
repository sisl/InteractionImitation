import gym
from core.gail import gail_ppo, Buffer
from core.value import Value
from core.policy import Policy
from core.discriminator import Discriminator
import torch.optim

env_fn = lambda _: gym.make('Pendulum-v0')

policy = Policy(env_fn(0).action_space.shape[0])
pi_opt = torch.optim.RMSprop(policy.parameters(), lr=3e-4)

value = Value()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3)

discriminator = Discriminator()
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-3)

expert_data = torch.load('trpo-pendulum-expert-data.pt')
expert_data = Buffer(*expert_data)

gail_ppo(
    env_fn=env_fn,
    expert_data=expert_data,
    discriminator=discriminator,
    disc_opt=disc_opt,
    disc_iters=10,
    policy=policy,
    value=value,
    v_opt=v_opt,
    v_iters=1000,
    epochs=100,
    rollout_episodes=20,
    rollout_steps=200,
    gamma=0.99,
    gae_lambda=0.9,
    clip_ratio=0.2,
    pi_opt=pi_opt,
    pi_iters=100,
    wasserstein=True,
    wasserstein_c=100.,
)

torch.save(policy.state_dict(), 'gail-pendulum.pt')