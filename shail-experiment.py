# %%
import os

import gym
from src.safe_options.options import gail_ppo, Buffer
from src.core.value import SetValue
from src.safe_options.policy import SetMaskedDiscretePolicy
from src.core.discriminator import DeepsetDiscriminator
import torch

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

DIR = os.path.dirname(os.path.abspath(__file__))
option_list = [[(vel, time) for vel in [0, 1, 2, 4, 6, 8, 10] for time in [5]],
                [(vel, time) for vel in [0, 1, 2, 5, 7.5, 10] for time in [5, 10]],
                [(vel, time) for vel in [0, 3, 10] for time in [5, 10, 20]]
]
activations = [torch.nn.Tanh, torch.nn.LeakyReLU]

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

def training_function(config):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    if config['experiment'] == 'A':
        envs = [SafeOptionsEnv(Setobs(
            TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
                n_rays=5,
                reward=functools.partial(
                    speed_reward,
                    collision_penalty=0
                ),
                check_collisions=True,
                stop_on_collision=config['trainenv']['stop_on_collision'],
            ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
            ), options=option_list[config['policy']['option']],
            safe_actions_collision_method=config['trainenv']['safe_actions_collision_method'], 
            abort_unsafe_collision_method=config['trainenv']['abort_unsafe_collision_method'],
            ) for _ in range(120)]

    elif config['experiment'] == 'B': 
        envs = sum([[SafeOptionsEnv(Setobs(
            TransformObservation(CollisionPenaltyWrapper(IntersimpleLidarFlatRandom(
                n_rays=5,
                reward=functools.partial(
                    speed_reward,
                    collision_penalty=0
                ),
                check_collisions=True,
                stop_on_collision=config['trainenv']['stop_on_collision'], track=track,
            ), collision_distance=6, collision_penalty=100), lambda obs: (obs - obs_min) / (obs_max - obs_min + 1e-10))
            ), options=option_list[config['policy']['option']],
            safe_actions_collision_method=config['trainenv']['safe_actions_collision_method'], 
            abort_unsafe_collision_method=config['trainenv']['abort_unsafe_collision_method'],
            ) for _ in range(15)] for track in range(4)],[])
    
    else:
        raise NotImplementedError

    env_fn = lambda i: envs[i]

    policy = SetMaskedDiscretePolicy(env_fn(0).action_space.n, 
        n_hidden_layers=config['policy']['n_hidden_layers'],
        hidden_layer_size=config['policy']['hidden_layer_size'],
        activation=activations[config['policy']['activation']] ) # config net architecture
    pi_opt = torch.optim.Adam(policy.parameters(), lr=config['policy']['learning_rate'])
    pi_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_opt, gamma=config['policy']['learning_rate_decay'])

    value = SetValue() # config net architecture
    v_opt = torch.optim.Adam(value.parameters(), lr=config['value']['learning_rate'])

    discriminator = DeepsetDiscriminator(
        n_hidden_layers_element=config['discriminator']['n_hidden_layers_element'],
        n_hidden_layers_global=config['discriminator']['n_hidden_layers_global'],
        hidden_layer_size=config['discriminator']['hidden_layer_size'],
        activation=activations[config['discriminator']['activation']],
    )
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=config['discriminator']['learning_rate'], weight_decay=config['discriminator']['weight_decay'])

    if config['experiment'] == 'A': 
        expert_data = torch.load(os.path.join(DIR, 'intersimple-expert-data-setobs2-loc0-track0.pt'))
    elif config['experiment'] == 'B': 
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

    def callback(info):
        tune.report(gen_mean_reward_per_episode=info['gen/mean_reward_per_episode'],
                    disc_mean_reward_per_episode=info['disc/mean_reward_per_episode'], 
                    mean_episode_length=info['gen/mean_episode_length'],
                    gen_collision_rate=info['gen/collision_rate'])
        
        # save model checkpoints
        ep = info['epoch'] + 1
        if (ep % 25 == 0):
            torch.save(info['policy'].state_dict(), f'policy_epoch{ep}.pt')

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
        epochs=config['train_epochs'],
        rollout_episodes=120, 
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

    # save model
    torch.save(policy.state_dict(), 'policy_final.pt')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', choices=['A', 'B'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--test', type=str, help='path to config file to run final training on')
    parser.add_argument('--test_seeds', type=int, default=5)
    parser.add_argument('--test_cpus', type=int, help='number of cpus available to split test seed training over')
    args = parser.parse_args()

    assert (args.train is None) ^ (args.test is None), 'Must either train on an experiment or test with a config file'

    # if no test config specified, train
    if args.test is None:
        print('Running Tuning for Experiment %s'%(args.train))
        analysis = tune.run(
            training_function,
            config={
                'experiment': args.train,
                'trainenv': {
                    'stop_on_collision': False, 
                    'safe_actions_collision_method': 'circle', 
                    'abort_unsafe_collision_method': 'circle',
                },
                'policy': {
                    'learning_rate': 3e-4, 
                    'learning_rate_decay': 1.0, 
                    'clip_ratio': 0.2, 
                    'iterations_per_epoch': tune.grid_search([250, 500, 750]),
                    'hidden_layer_size': 40, #tune.grid_search([20, 40]),
                    'n_hidden_layers': 3, #tune.grid_search([2, 3]), 
                    'activation':0,
                    'option': 0, #tune.grid_search(list(range(len(option_list))))
                },
                'value': {
                    'learning_rate': 1e-3, 
                    'iterations_per_epoch': 1000, 
                },
                'discriminator': {
                    'learning_rate': 1e-3, 
                    'weight_decay': 1e-4, 
                    'iterations_per_epoch': 100, 
                    'n_hidden_layers_element': 3, #tune.grid_search([3,4]),
                    'n_hidden_layers_global': 2, #tune.grid_search([1,2]),
                    'hidden_layer_size': 10, 
                    'activation': 0,
                },
                'train_epochs': args.epochs,
                'seed': 0,
            }
        )
        best_config = analysis.get_best_config(metric='gen_collision_rate', mode='min')
        print('Best config: ', best_config)

        # safe best_config
        if not os.path.isdir(os.path.join(DIR, 'best_configs')):
            os.mkdir(os.path.join(DIR, 'best_configs'))
        
        # save shail
        with open(os.path.join(DIR, 'best_configs',f'shail_exp{args.train}.json'), 'w', encoding='utf-8') as f:
            json.dump(best_config, f, ensure_ascii=False, indent=4)

        # save hail
        best_config['trainenv']['safe_actions_collision_method']=None
        best_config['trainenv']['abort_unsafe_collision_method']=None
        with open(os.path.join(DIR, 'best_configs',f'hail_exp{args.train}.json'), 'w', encoding='utf-8') as f:
            json.dump(best_config, f, ensure_ascii=False, indent=4)
    
    # if config file specified, rerun it with appropriate number of seeds
    else:
        with open(args.test, 'rb') as f:
            config = json.load(f)

        print(f'Retraining {args.test} with {args.test_seeds} seeds on experiment {config["experiment"]}')

        # rerun with appropriate number of seeds
        rpt = {'cpu': int(args.test_cpus/args.test_seeds)} if (args.test_cpus is not None) else None
        config['seed'] = tune.grid_search(list(range(1,args.test_seeds+1)))
        analysis = tune.run(training_function, config=config, resources_per_trial=rpt)

        # move final policies to appropriate directory
        split_ = os.path.basename(args.test).split('_')
        model = split_[0]
        exper = split_[-1].split('.')[0]
        savepath = os.path.join('test_policies',model,exper)
        
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        
        import shutil

        # save config
        shutil.copyfile(
            args.test,
            os.path.join(savepath, 'config.json')
        )

        for i in range(args.test_seeds):
            s = analysis._checkpoints[i]['config']['seed']
            check_dir = analysis._checkpoints[i]['logdir']
            shutil.copyfile(os.path.join(check_dir,'policy_final.pt'), 
                os.path.join(savepath, f'policy_seed{s}.pt'))