# %%
import gym
from core.sampling import rollout
from core.trpo import trpo
from core.value import Value
from core.policy import Policy
import torch.optim
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools
import numpy as np
from gym.wrappers import TransformObservation
from util.wrappers import CollisionPenaltyWrapper
from core.reparam_module import ReparamPolicy

from util.wrappers import Minobs

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

envs = [Minobs(TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
    n_rays=5,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
    stop_on_collision=False,
), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))) for _ in range(50)]

env_fn = lambda i: envs[i]
policy = Policy(env_fn(0).action_space.shape[0])
value = Value()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-4, weight_decay=1e-3)

# %%
value, policy = trpo(
    env_fn=env_fn,
    value=value,
    policy=policy,
    epochs=200,
    rollout_episodes=30,
    rollout_steps=100,
    gamma=0.99,
    gae_lambda=0.95,
    delta=0.01,
    backtrack_coeff=0.9,
    backtrack_iters=50,
    v_opt=v_opt,
    v_iters=1000,
    cg_damping=0.1,
)

torch.save(policy.state_dict(), 'trpo-intersimple-minobs2.pt')

# %%
policy = Policy(env_fn(0).action_space.shape[0])
policy(torch.zeros(env_fn(0).observation_space.shape))
policy = ReparamPolicy(policy)
policy.load_state_dict(torch.load('trpo-intersimple-minobs2.pt'))

env = env_fn(0)
obs = env.reset()
env.render(mode='post')
for i in range(300):
    #action, _ = policy.predict(torch.tensor(obs))
    action = policy.sample(policy(torch.tensor(obs, dtype=torch.float32)))
    obs, reward, done, _ = env.step(action)
    env.render(mode='post')
    print('step', i, 'reward', reward)
    if done:
        break
env.close()
