import os
import torch
import gym
import intersim
import numpy as np
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

def main(config, method='bc', train=False, test=False, loc=0, datadir='./expert_data', filestr='', **kwargs):
    """
    Main loop for training and testing different imitation models
    Args:
        config (dict): configuration dictionary for model
        train (bool): whether to run train loop
        test (bool): whether to run test loop
        method (str): the method to try for imitation
        loc (int): the location index of the roundabout
        datadir (str): path to expert data
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
