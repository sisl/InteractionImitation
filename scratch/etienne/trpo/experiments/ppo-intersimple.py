from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools

from core.ppo import ppo
from core.value import Value
from core.policy import Policy
import torch.optim

envs = [IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=10
    ),
) for _ in range(30)]

env_fn = lambda i: envs[i]
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3)

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
