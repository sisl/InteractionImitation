# %%
import gym
from options.options import gail
from core.gail import Buffer
from core.value import SetValue
from core.policy import SetDiscretePolicy
from core.discriminator import DeepsetDiscriminator
import torch.optim
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools
from util.wrappers import CollisionPenaltyWrapper, TransformObservation, Setobs
import numpy as np
from options.options import OptionsEnv
from torch.utils.tensorboard import SummaryWriter
from core.reparam_module import ReparamPolicy

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

envs = [OptionsEnv(Setobs(
    TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
        n_rays=5,
        reward=functools.partial(
            speed_reward,
            collision_penalty=0
        ),
        stop_on_collision=False,
    ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
), options=[(0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (8, 5)]) for _ in range(60)]

env_fn = lambda i: envs[i]
policy = SetDiscretePolicy(env_fn(0).action_space.n)
value = SetValue()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-4)

discriminator = DeepsetDiscriminator()
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)

expert_data = torch.load('intersimple-expert-data-setobs2.pt')
expert_data = Buffer(*expert_data)

# %%
value, policy = gail(
    env_fn=env_fn,
    expert_data=expert_data,
    discriminator=discriminator,
    disc_opt=disc_opt,
    disc_iters=100,
    policy=policy,
    value=value,
    v_opt=v_opt,
    v_iters=1000,
    epochs=200,
    rollout_episodes=60,
    rollout_steps=60,
    gamma=0.99,
    gae_lambda=0.9,
    delta=0.01,
    backtrack_coeff=0.8,
    backtrack_iters=10,
    wasserstein=True,
    wasserstein_c=1.,
    logger=SummaryWriter(comment='wgail-options-setobs2'),
)

torch.save(policy.state_dict(), 'wgail-options-setobs2.pt')

# %%
policy = SetDiscretePolicy(env_fn(0).action_space.n)
policy(torch.zeros(env_fn(0).observation_space.shape))
policy = ReparamPolicy(policy)
policy.load_state_dict(torch.load('wgail-options-setobs2.pt'))

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
