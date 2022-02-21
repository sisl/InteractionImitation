import gym
from gym.wrappers import TransformObservation
from core.trpo import trpo
from core.value import Value
from core.policy import Policy
import torch.optim

env_fn = lambda _: TransformObservation(gym.make('Pendulum-v0'), lambda obs: obs)
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3, weight_decay=1e-4)

value, policy = trpo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=100,
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


torch.save(policy.state_dict(), 'trpo-pendulum.pt')
