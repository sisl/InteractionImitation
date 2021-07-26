import torch
import gym
import intersim
import numpy as np
import json5
import os
opj = os.path.join
from tqdm import tqdm

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

def main(method='bc', train=False, test=False, loc=0, config_path=None, **kwargs):
    """
    Main loop for training and testing different imitation models
    Args:
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

    # make prefix of output files
    outdir = opj('output',method,'loc%02i'%(loc))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    filestr = opj(outdir, basestr(**kwargs))
    
    # load config
    if config_path:
        with open(config_path, 'r') as cfg:
            config = json5.load(cfg)
    else:
        raise Exception('No config path specified')

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
        train_dataset = InteractionDatasetSingleAgent(loc=loc, tracks=[0,1,2])
        cv_dataset = InteractionDatasetSingleAgent(loc=loc, tracks=[3])
        train_fn(train_dataset, cv_dataset, policy, filestr, **kwargs)
    
    if test:

        # load policy 
        policy = policy_class.load_model(config, filestr)
        policy.eval()

        # simulate policy
        track = 4
        simulate_policy(policy, loc=loc, track=track, filestr=filestr, nframes=500)

        # run test metrics
        test_dataset = InteractionDatasetSingleAgent(loc=loc, tracks=[track])
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
    svt, svt_path = get_svt(base='InteractionSimulator', loc=loc, track=track)
    osm = get_map_path(base='InteractionSimulator', loc=loc)
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
    parser.add_argument("--test", help="test model",
        action="store_true") 
    parser.add_argument("--method", help="modeling method",
        choices=['bc', 'gail', 'advil'], default='bc')
    parser.add_argument("--config", help="config file path",
        default='config/networks.json5', type=str)
    parser.add_argument('--seed', default=0, type=int,
        help='seed')    
    args = parser.parse_args()
    kwargs = {
        'train':args.train, 
        'test':args.test, 
        'method':args.method,
        'loc':args.loc,
        'config_path':args.config,
        'seed':args.seed
    }
    return kwargs


if __name__ == '__main__':
    kwargs = parse_args()
    main(**kwargs)