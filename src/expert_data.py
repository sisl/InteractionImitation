import torch

import pickle
import gym
import numpy as np

import intersim
from intersim.utils import get_map_path, get_svt, SVT_to_sim_stateactions

import os
opj = os.path.join

def generate_expert_data(path: str='expert_data', loc: int = 0, track:int = 0, **kwargs):
    """
    Function to save (joint) states and observations from simulated frame
    Args:
        path (str): directory to save data
        loc (int): location index
        track (int): track index
        kwargs: arguments for environment instantiation
    """
    
    if not os.path.isdir(path):
        os.mkdir(path)
    filestr = opj(path,intersim.LOCATIONS[loc]+'_track%03i'%(track))
    
    svt, svt_path = get_svt(base='InteractionSimulator', loc=loc, track=track)
    osm = get_map_path(base='InteractionSimulator', loc=loc)
    print('SVT path: {}'.format(svt_path))
    print('Map path: {}'.format(osm))
    states, actions = SVT_to_sim_stateactions(svt)

    # animate from environment
    env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, **kwargs, 
        min_acc=-np.inf, max_acc=np.inf)
    
    env.reset()
    done = False
    obs, actions_taken = [], []
    i = 0
    print(actions.shape)
    while not done:
        ob, r, done, info = env.step(actions[i])
        obs.append(ob)
        actions_taken.append(info['action_taken'])
        i += 1
    
    # shift actions
    actions_taken.pop(0)
    obs.pop(-1)

    # save observations and actions
    pickle.dump(obs,open(filestr+'_observations.pkl', 'wb'))
    torch.save(torch.stack(actions_taken), filestr+'_actions.pt')

def load_expert_data(path='expert_data', loc: int = 0, track:int = 0):
    """
    Load expert data from file.
    Args:
        path (str): directory to save data
        loc (int): location index
        track (int): track index
    Returns:
        obs (list[Observations]): list of observations
        actions (list[torch.tensor]): list of corresponding actions taken in observations
    """
    # load observations and actions
    filestr = opj(path, intersim.LOCATIONS[loc]+'_track%03i'%(track))
    obs = pickle.load(open(filestr+'_observations.pkl', 'rb'))
    actions = torch.load(filestr+'_actions.pt')
    actions = list(torch.unbind(actions))
    return obs, actions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Save Expert Trajectories')
    parser.add_argument('--loc', default=0, type=int,
                       help='location (default 0)')
    parser.add_argument('--track', default=0, type=int,
                       help='track number (default 0)')
    args = parser.parse_args()
    generate_expert_data(loc=args.loc,track=args.track)