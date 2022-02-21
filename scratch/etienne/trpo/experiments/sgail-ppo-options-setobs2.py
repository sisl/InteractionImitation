# %%
import sys
sys.path.append('../../../../')

import gym
from src.safe_options.options import gail_ppo, Buffer
from src.core.value import SetValue
from src.safe_options.policy import SetMaskedDiscretePolicy
from src.core.discriminator import DeepsetDiscriminator
import torch.optim
from intersim.envs import IntersimpleLidarFlatRandom
from intersim.envs.intersimple import speed_reward
import functools
from src.util.wrappers import CollisionPenaltyWrapper, TransformObservation, Setobs
import numpy as np
from src.safe_options.options import SafeOptionsEnv
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

envs = [SafeOptionsEnv(Setobs(
    TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
        n_rays=5,
        reward=functools.partial(
            speed_reward,
            collision_penalty=0
        ),
        stop_on_collision=True,
    ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
), options=[(0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (8, 5), (10, 5)], safe_actions_collision_method='circle', abort_unsafe_collision_method='circle') for _ in range(60)]

env_fn = lambda i: envs[i]

policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n)
pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

value = SetValue()
v_opt = torch.optim.Adam(value.parameters(), lr=1e-3)

discriminator = DeepsetDiscriminator()
disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-4)

expert_data = torch.load('intersimple-expert-data-setobs2.pt')
expert_data = Buffer(*expert_data)

# %%
def callback(epoch, value, policy):
    if not epoch % 10:
        torch.save(policy.state_dict(), f'sgail-ppo-options-setobs2-{epoch}.pt')
        torch.save(value.state_dict(), f'sgail-ppo-options-setobs2-value-{epoch}.pt')

value, policy = gail_ppo(
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
    clip_ratio=0.2,
    pi_opt=pi_opt,
    pi_iters=100,
    logger=SummaryWriter(comment='sgail-ppo-options-setobs2'),
    callback=callback,
)

torch.save(policy.state_dict(), 'sgail-ppo-options-setobs2.pt')

# %%
policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n)
policy(torch.zeros(env_fn(0).observation_space['observation'].shape), torch.zeros(env_fn(0).observation_space['safe_actions'].shape))
policy.load_state_dict(torch.load('sgail-ppo-options-setobs2.pt'))

env = env_fn(0)
obs = env.reset()
env.render(mode='post')
for i in range(300):
    action = policy.sample(policy(
        torch.tensor(obs['observation'], dtype=torch.float32),
        torch.tensor(obs['safe_actions'], dtype=torch.float32),
    ))
    obs, reward, done, _ = env.step(action, render_mode='post')
    print('step', i, 'reward', reward, 'safe actions', obs['safe_actions'])
    if done:
        break
env.close()
# %%
