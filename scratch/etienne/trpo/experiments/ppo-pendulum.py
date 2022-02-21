import gym
from core.ppo import ppo
from core.value import Value
from core.policy import Policy
import torch.optim

env_fn = lambda _: gym.make('Pendulum-v0')
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3)

ppo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=300,
    rollout_episodes=100,
    rollout_steps=200,
    gamma=0.99,
    gae_lambda=0.9,
    clip_ratio=0.2,
    pi_opt=pi_opt,
    pi_iters=100,
    v_opt=v_opt,
    v_iters=1000,
)
