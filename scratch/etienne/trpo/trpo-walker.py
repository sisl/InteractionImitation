import gym
from core.trpo import trpo
from core.value import Value
from core.policy import Policy
import torch.optim

env_fn = lambda _: gym.make('BipedalWalker-v3')
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-2)

trpo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=1000,
    rollout_episodes=20,
    rollout_steps=250,
    gamma=0.99,
    gae_lambda=0.9,
    delta=0.01,
    backtrack_coeff=0.8,
    backtrack_iters=10,
    v_opt=v_opt,
    v_iters=1000,
)
