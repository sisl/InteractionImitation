# %%
import gym
from core.sampling import rollout
from core.ppo import ppo
from core.value import Value
from core.policy import DiscretePolicy
import torch.optim
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools
import numpy as np
from util.wrappers import CollisionPenaltyWrapper, TransformObservation

from util.wrappers import Minobs
from options.options import OptionsEnv

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

envs = [OptionsEnv(Minobs(
    TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlat(
        n_rays=5,
        agent=51,
        reward=functools.partial(
            speed_reward,
            collision_penalty=0
        ),
        stop_on_collision=False,
    ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
), options=[(0, 5), (5, 5), (10, 5)]) for _ in range(50)]

env_fn = lambda i: envs[i]
policy = DiscretePolicy(env_fn(0).action_space.n)
value = Value()
pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3)

# %%
value, policy = ppo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=50,
    rollout_episodes=30,
    rollout_steps=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    pi_opt=pi_opt,
    pi_iters=100,
    v_opt=v_opt,
    v_iters=1000,
)

torch.save(policy.state_dict(), 'ppo-options-minobs.pt')

# %%
policy = DiscretePolicy(env_fn(0).action_space.n)
policy(torch.zeros(env_fn(0).observation_space.shape))
policy.load_state_dict(torch.load('ppo-options-minobs.pt'))

env = env_fn(0)
obs = env.reset()
env.render(mode='post')
for i in range(300):
    #action, _ = policy.predict(torch.tensor(obs))
    action = policy.sample(policy(torch.tensor(obs, dtype=torch.float32)))
    obs, reward, done, _ = env.step(action, render_mode='post')
    print('step', i, 'reward', reward)
    if done:
        break
env.close()

# %%