import torch
import gym
import intersim
import numpy as np
import json5
import os
opj = os.path.join
from tqdm import tqdm
from functools import partial

from src import InteractionDatasetSingleAgent, metrics
from intersim.utils import get_map_path, get_svt

def basestr(**kwargs):
    """
    Return base prefix for all files relating to a certain experiment
    Args:
        kwargs (dict): keyword arguments sent to main training loop
    Returns:
        basestr (str): prefix
    """
    return 'base'

def main(config, method='bc', train=False, test=False, loc=0, datadir='./expert_data', filestr='', **kwargs):
    """
    Main loop for training and testing different imitation models
    Args:
        config (dict): configuration dictionary for model
        train (bool): whether to run train loop
        test (bool): whether to run test loop
        method (str): the method to try for imitation
        loc (int): the location index of the roundabout
        config_file (str): path to config file
        kwargs (dict): remaining kwargs for training loop
    """
    # get/set seed
    seed = kwargs.get('seed',0)
    torch.manual_seed(seed)

    # method-based training
    if method=='bc':
        from src import bc
        policy_class = bc.BehaviorCloningPolicy
        train_fn = bc.train
    else:
        raise NotImplementedError
    
    # default train / cv / test split datasets 
    if train:

        # make policy, train and test datasets, and send to 
        policy = policy_class(config)
        train_dataset = InteractionDatasetSingleAgent(output_dir=datadir, loc=loc, tracks=[0])#,1,2])
        cv_dataset = InteractionDatasetSingleAgent(output_dir=datadir, loc=loc, tracks=[3])
        train_fn(config, policy, train_dataset, cv_dataset, filestr, **kwargs)
    
    if test:

        # load policy 
        policy = policy_class.load_model(config, filestr)
        policy.eval()

        # simulate policy
        track = 4
        simulate_policy(policy, loc=loc, track=track, filestr=filestr, nframes=500)

        # run test metrics
        test_dataset = InteractionDatasetSingleAgent(output_dir=datadir, loc=loc, tracks=[track])
        metrics(filestr, test_dataset, policy)


def simulate_policy(policy, loc=0, track=0, filestr='', nframes=float('inf')):
    """
    Simulate a trained policy
    Args:
        policy: the policy to simulate, which should return action directly
        loc (int): location index to test policy
        track (int): track to test policy
        filestr (str): path prefix to save simulation to
    """
    # animate from environment
    basepath = os.path.abspath('./InteractionSimulator')
    svt, svt_path = get_svt(base=basepath, loc=loc, track=track)
    osm = get_map_path(base=basepath, loc=loc)
    env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, 
        min_acc=-np.inf, max_acc=np.inf)
    # env = gym.make('intersim:intersim-v0', loc=loc, track=track, 
    #     min_acc=-np.inf, max_acc=np.inf)
    
    ob, _ = env.reset()
    env.render()
    done = False
    i = 0
    with tqdm(total=min(nframes, env._svt.Tind)) as pbar:
        while not done and i < nframes:  
            i += 1
            
            # get action
            action = policy(ob)

            # propagate environment
            ob, r, done, info = env.step(action)
            env.render()

            pbar.update()

    env.close(filestr=filestr)

def parse_args():
    """
    Parse arguments to main
    Returns:
        kwargs: dictionary of arguments:
            train (bool): whether to run train loop
            test (bool): whether to run test loop
            method (str): the method to try for imitation
            loc (int): the location index of the roundabout
            config (str): config path
            seed (int): RNG seed
    """
    import argparse
    parser = argparse.ArgumentParser(description='Save Expert Trajectories')
    parser.add_argument('--loc', default=0, type=int,
        help='location (default 0)')
    parser.add_argument("--train", help="train model",
        action="store_true")
    parser.add_argument("--all-runs", help="use ray tune to run multiple experiments",
        action="store_true")
    parser.add_argument("--test", help="test model",
        action="store_true") 
    parser.add_argument("--method", help="modeling method",
        choices=['bc', 'gail', 'advil'], default='bc')
    parser.add_argument("--config", help="config file path",
        default=None, type=str)
    parser.add_argument('--seed', default=0, type=int,
        help='seed')    
    args = parser.parse_args()
    kwargs = {
        'train':args.train, 
        'test':args.test, 
        'method':args.method,
        'loc':args.loc,
        'config_path':args.config,
        'seed':args.seed,
        'all_runs':args.all_runs
    }
    return kwargs

def main_wrapper(**kwargs):
    if kwargs['all_runs']:
        pass
    else:
        main(**kwargs)

def get_full_config(ray_config:dict, method:str)->dict:
    """
    Get full model configuration from ray config and method string
    Args:
        ray_config (dict): ray config
        method (str): method to get full configuration for
    """
    if method == 'bc':
        from src.bc import bc_config
        config = bc_config(ray_config)
    else:
        raise NotImplementedError
    return config

def get_ray_config(method:str)->dict:
    """
    Get configuration for ray based on method.
    Args: 
        method (str): method to get configuration for
    Returns:
        ray_config (dict): configuration for ray
    """
    if method == 'bc':
        ray_config = {
            "lr": tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": tune.choice([0.001, 0.01, 0.1, 0.5, 0.9]),
            "loss": tune.choice(['huber', 'mse']),
            "train_batch_size": tune.choice([16,32,64]),
            "deepsets_phi_hidden_n": tune.choice([1,2,3]),
            "deepsets_phi_hidden_dim": tune.choice([16,32,64]),
            "deepsets_latent_dim": tune.choice([16,32,64]),
            "deepsets_rho_hidden_n": tune.choice([0,1,2]),
            "deepsets_rho_hidden_dim": tune.choice([16,32,64]),
            "deepsets_output_dim": tune.choice([8,16,32,64]),
            "head_hidden_n": tune.choice([0,1,2]),
            "head_hidden_dim": tune.choice([16,32,64]),
            "head_final_activation": tune.choice(['sigmoid', None]),
        }
    else:
        raise NotImplementedError
    return ray_config

if __name__ == '__main__':
    kwargs = parse_args()
    
    # make prefix of output files
    outdir = opj('output',kwargs['method'],'loc%02i'%(kwargs['loc']))

    if kwargs['config_path']:
        # load config
        with open(kwargs['config_path'], 'r') as cfg:
            config = json5.load(cfg)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)   
        filestr = opj(outdir, basestr(**kwargs)) 
        main(config, filestr=filestr, **kwargs)

    elif kwargs['all_runs'] and kwargs['train']:
        
        def ray_train(config, datadir=None):
            full_config = get_full_config(config, kwargs['method'])
            main(full_config, filestr='exp', datadir=datadir, ray=True, **kwargs)
        
        # set up ray tune
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        datadir = os.path.abspath('./expert_data')
        ray_config = get_ray_config(kwargs['method'])
        custom_scheduler = ASHAScheduler(
            metric='cv_loss',
            mode="min",
            grace_period=25,
        )
        analysis = tune.run(
            partial(ray_train, datadir=datadir),
            config=ray_config,
            scheduler=custom_scheduler,
            local_dir=outdir,
            resources_per_trial={"cpu": 2},
            num_samples=20,
        )
    else:
        raise Exception('No valid config found')

    
    
    

    