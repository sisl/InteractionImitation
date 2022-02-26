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
from datetime import datetime
import json

def training_function(config):
    DIR = os.path.dirname(os.path.abspath(__file__))
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
            check_collisions=True,
            stop_on_collision=config['env']['stop_on_collision'],
        ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
    ), options=[(0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (8, 5), (10, 5)],
        safe_actions_collision_method=config['env']['safe_actions_collision_method'],
        abort_unsafe_collision_method=config['env']['abort_unsafe_collision_method']) for _ in range(60)]

    env_fn = lambda i: envs[i]

    policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n, hidden_layer_size=config['policy']['hidden_layer_size']) # config net architecture
    pi_opt = torch.optim.Adam(policy.parameters(), lr=config['policy']['learning_rate'])
    pi_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_opt, gamma=config['policy']['learning_rate_decay'])

    value = SetValue() # config net architecture
    v_opt = torch.optim.Adam(value.parameters(), lr=config['value']['learning_rate'])

    discriminator = DeepsetDiscriminator() # config net architecture
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config['discriminator']['learning_rate'], weight_decay=config['discriminator']['weight_decay'])

    expert_data = [
        torch.load(os.path.join(DIR, 'intersimple-expert-data-setobs2-loc0-track0.pt')),
        torch.load(os.path.join(DIR, 'intersimple-expert-data-setobs2-loc0-track1.pt')),
        torch.load(os.path.join(DIR, 'intersimple-expert-data-setobs2-loc0-track2.pt')),
        torch.load(os.path.join(DIR, 'intersimple-expert-data-setobs2-loc0-track3.pt')),
    ]
    d0 = [d[0] for d in expert_data]
    d1 = [d[1] for d in expert_data]
    d2 = [d[2] for d in expert_data]
    d3 = [d[3] for d in expert_data]
    expert_data = (torch.cat(d0), torch.cat(d1), torch.cat(d2), torch.cat(d3))
    expert_data = Buffer(*expert_data)

    folder = str(datetime.now())
    os.mkdir(os.path.join(DIR, folder))
    with open(os.path.join(DIR, folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    def callback(info):
        tune.report(gen_mean_reward_per_episode=info['gen/mean_reward_per_episode'])
        if not info['epoch'] % 10:
            torch.save(policy.state_dict(), os.path.join(DIR, folder, f'sgail-ppo-options-setobs2-{info["epoch"]}.pt'))
            torch.save(value.state_dict(), os.path.join(DIR, folder, f'sgail-ppo-options-setobs2-value-{info["epoch"]}.pt'))

    value, policy = gail_ppo(
        env_fn=env_fn,
        expert_data=expert_data,
        discriminator=discriminator,
        disc_opt=disc_opt,
        disc_iters=config['discriminator']['iterations_per_epoch'],
        policy=policy,
        value=value,
        v_opt=v_opt,
        v_iters=config['value']['iterations_per_epoch'],
        epochs=301,
        rollout_episodes=60,
        rollout_steps=60,
        gamma=0.99,
        gae_lambda=0.9,
        clip_ratio=config['policy']['clip_ratio'],
        pi_opt=pi_opt,
        pi_iters=config['policy']['iterations_per_epoch'],
        logger=SummaryWriter(comment='sgail-ppo-options-setobs2'),
        callback=callback,
        lr_schedulers=[pi_lr_scheduler],
    )

analysis = tune.run(
    training_function,
    config={
        'env': {
            'stop_on_collision': False,
            'safe_actions_collision_method': 'circle',
            'abort_unsafe_collision_method': 'circle',
        },
        'policy': {
            'learning_rate': tune.grid_search([3e-4]),
            'learning_rate_decay': tune.grid_search([1.0]),
            'clip_ratio': tune.grid_search([0.2]),
            'iterations_per_epoch': tune.grid_search([100]),
            'hidden_layer_size': tune.grid_search([25])
        },
        'value': {
            'learning_rate': tune.grid_search([1e-3]),
            'iterations_per_epoch': tune.grid_search([1000]),
        },
        'discriminator': {
            'learning_rate': tune.grid_search([1e-3]),
            'weight_decay': tune.grid_search([1e-4]),
            'iterations_per_epoch': tune.grid_search([500]),
        }
    }
)

print('Best config: ', analysis.get_best_config(metric='gen_mean_reward_per_episode', mode='max'))

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
