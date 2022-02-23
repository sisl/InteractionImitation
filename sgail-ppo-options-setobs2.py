# %%
import os

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
from ray import tune

def training_function(config):
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

    policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n) # config net architecture
    pi_opt = torch.optim.Adam(policy.parameters(), lr=3e-5)  # config learning rate
    pi_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_opt, gamma=0.98) # config lr decay

    value = SetValue() # config net architecture
    v_opt = torch.optim.Adam(value.parameters(), lr=1e-3) # config lr

    discriminator = DeepsetDiscriminator() # config net architecture
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3, weight_decay=1e-3) # config lr, weight decay

    expert_data = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'intersimple-expert-data-setobs2.pt'))
    expert_data = Buffer(*expert_data)

    def callback(info):
        tune.report(gen_mean_reward_per_episode=info['gen/mean_reward_per_episode'])

    value, policy = gail_ppo(
        env_fn=env_fn,
        expert_data=expert_data,
        discriminator=discriminator,
        disc_opt=disc_opt,
        disc_iters=100, # config
        policy=policy,
        value=value,
        v_opt=v_opt,
        v_iters=1000, # config
        epochs=200,
        rollout_episodes=60,
        rollout_steps=60,
        gamma=0.99,
        gae_lambda=0.9,
        clip_ratio=0.2, # config
        pi_opt=pi_opt,
        pi_iters=100, # config
        logger=SummaryWriter(comment='sgail-ppo-options-setobs2'),
        callback=callback,
        lr_schedulers=[pi_lr_scheduler],
    )

analysis = tune.run(
    training_function,
    config={
        'dummy': tune.grid_search([0.001, 0.01, 0.1]),
    }
)

print('Best config: ', analysis.get_best_config(metric='gen_mean_reward_per_episode', mode='min'))

# %%
# policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n)
# policy(torch.zeros(env_fn(0).observation_space['observation'].shape), torch.zeros(env_fn(0).observation_space['safe_actions'].shape))
# policy.load_state_dict(torch.load('sgail-ppo-options-setobs2.pt'))

# env = env_fn(0)
# obs = env.reset()
# env.render(mode='post')
# for i in range(300):
#     action = policy.sample(policy(
#         torch.tensor(obs['observation'], dtype=torch.float32),
#         torch.tensor(obs['safe_actions'], dtype=torch.float32),
#     ))
#     obs, reward, done, _ = env.step(action, render_mode='post')
#     print('step', i, 'reward', reward, 'safe actions', obs['safe_actions'])
#     if done:
#         break
# env.close()
# %%
