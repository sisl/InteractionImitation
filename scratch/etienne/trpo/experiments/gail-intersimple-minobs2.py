# %%
import gym
from core.gail import gail, Buffer
from core.value import Value
from core.policy import Policy
from core.discriminator import Discriminator
import torch.optim
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools
from util.wrappers import CollisionPenaltyWrapper, Minobs
import numpy as np
from gym.wrappers import TransformObservation
from core.reparam_module import ReparamPolicy
from torch.utils.tensorboard import SummaryWriter

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

discriminator = Discriminator()
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-5)

expert_data = torch.load('intersimple-expert-data-minobs2.pt')
expert_data = Buffer(*expert_data)

# %%
value, policy = gail(
    env_fn=env_fn,
    expert_data=expert_data,
    discriminator=discriminator,
    disc_opt=disc_opt,
    disc_iters=500,
    policy=policy,
    value=value,
    v_opt=v_opt,
    v_iters=1000,
    epochs=800,
    rollout_episodes=50,
    rollout_steps=200,
    gamma=0.99,
    gae_lambda=0.9,
    delta=0.01,
    backtrack_coeff=0.8,
    backtrack_iters=10,
    logger=SummaryWriter(comment='minobs2'),
)

torch.save(policy.state_dict(), 'gail-intersimple-minobs2.pt')

# %%
policy = Policy(env_fn(0).action_space.shape[0])
policy(torch.zeros(env_fn(0).observation_space.shape))
policy = ReparamPolicy(policy)
policy.load_state_dict(torch.load('gail-intersimple-minobs2.pt'))

env = env_fn(0)
env.random_skip = False
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

# %%
