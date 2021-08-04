import torch

import pickle
import gym
import numpy as np

import intersim
from intersim.utils import get_map_path, get_svt, SVT_to_stateactions
from intersim import collisions
from intersim.graphs import ConeVisibilityGraph
import os
opj = os.path.join

def generate_expert_data(path: str='expert_data', loc: int = 0, track:int = 0, 
                            mask_relstate: bool = False, regularize_actions: bool = False, 
                            **kwargs):
    """
    Function to save (joint) states and observations from simulated frame
    Args:
        path (str): directory to save data
        loc (int): location index
        track (int): track index
        mask_relstate (bool): whether to mask the relative states from the cone visibility graph
        regularize_actions (bool): whether to regularize the action selection
        kwargs: arguments for environment instantiation
    """
    
    action_reg = 0.002 if regularize_actions else 0
    
    if not os.path.isdir(path):
        os.makedirs(path)
    filestr = opj(path,intersim.LOCATIONS[loc]+'_track%03i'%(track))
    
    svt, svt_path = get_svt(loc=loc, track=track) #base='InteractionSimulator'
    osm = get_map_path(loc=loc)
    print('SVT path: {}'.format(svt_path))
    print('Map path: {}'.format(osm))
    states, actions = SVT_to_stateactions(svt)

    # animate from environment
    if mask_relstate:
        cvg = ConeVisibilityGraph(r=20, half_angle=120)
        env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, 
            min_acc=-np.inf, max_acc=np.inf, graph=cvg, mask_relstate=True, **kwargs)
    else:
        env = gym.make('intersim:intersim-v0', svt=svt, map_path=osm, **kwargs, 
            min_acc=-np.inf, max_acc=np.inf)
    
    env.reset()
    done = False
    obs, actions_taken, max_devs = [], [], []
    i = 0
    while not done and i < len(actions):
        # check state deviation
        env_state = env.projected_state
        nni = ~torch.isnan(env_state[:,0])
        norms = torch.norm(env_state[nni,:2]-states[i,nni,:2], dim=1)
        if len(norms)>0:
            max_devs.append(norms.max())

        # propagate environment
        ob, r, done, info = env.step(env.target_state(svt.simstate[i+1], mu=action_reg))
        obs.append(ob)
        actions_taken.append(info['action_taken'])
        i += 1
    
    print("Maximum environment deviation from track: %f m" %(max(max_devs)))

    # check for collisions
    x = torch.stack([ob['state'] for ob in obs])
    cols = collisions.check_collisions_trajectory(x, svt.lengths, svt.widths)
    assert ~torch.any(cols), 'Error: Collisions found at indices {}'.format(cols.nonzero(as_tuple=True))

    # shift actions
    actions_taken.pop(0)
    obs.pop(-1)
    actions = torch.stack(actions_taken)

    # save observations and actions
    pickle.dump(obs,open(filestr+'_raw_observations.pkl', 'wb'))
    torch.save(actions, filestr+'_raw_actions.pt')
    process_expert_observations(obs, actions, filestr)

def process_expert_observations(obs, actions, filestr, remove_outliers=True, dtype=torch.float32):
    """
    Process the expert observations and save them as torch tensors
    Args:
        obs (list[dict]): lost of observations
        actions (torch.Tensor): (T, nv, a) tensor of actions
        filestr (str): base filename with which to save out observation tensors
        remove_outliers (bool): whether to remove datapoints with acceleration above or below 5 m/s/s
        dtype (torch.Type): type to convert data to
    """
    keys = ['ego_state', 'relative_state', 'path', 'action', 'next_ego_state', 'next_relative_state', 'next_path']
    data = {key:[] for key in keys}
    assert len(obs) == len(actions), 'non-matching action and observation lengths'
    T = len(obs)
    max_nv = 0
    for t in range(T-1):
        nni = ~torch.isnan(obs[t]['state'][:,0]) & ~torch.isnan(obs[t+1]['state'][:,0]) 
        max_nv = max(max_nv,nni.count_nonzero())

        # state
        data['ego_state'].append(obs[t]['state'][nni])
        data['relative_state'].append(obs[t]['relative_state'].index_select(0, 
            nni.nonzero()[:,0]).index_select(1, nni.nonzero()[:,0]))
        data['path'].append(torch.stack((obs[t]['paths'][0][nni], obs[t]['paths'][1][nni]), dim=-1))

        # action
        data['action'].append(actions[t][nni])

        # next state
        data['next_ego_state'].append(obs[t+1]['state'][nni])
        data['next_relative_state'].append(obs[t+1]['relative_state'].index_select(0, 
            nni.nonzero()[:,0]).index_select(1, nni.nonzero()[:,0]))
        data['next_path'].append(torch.stack((obs[t+1]['paths'][0][nni], obs[t+1]['paths'][1][nni]), dim=-1))

    

    # pad second dimension of relative state
    for i in range(len(data['relative_state'])):
        nv1, nv2, d = data['relative_state'][i].shape
        pad = torch.zeros(nv1, max_nv-nv2, d, dtype=dtype) * np.nan
        data['relative_state'][i] = torch.cat((data['relative_state'][i], pad), dim=1)
        data['next_relative_state'][i] = torch.cat((data['next_relative_state'][i], pad), dim=1)

    # cat lists 
    for key in keys:
        data[key] = torch.cat(data[key]).type(dtype)

    if remove_outliers:
        non_outlier_indices = torch.nonzero(torch.abs(data['action'][:,0]) < 5)
        for key in keys:
            data[key] = data[key][non_outlier_indices[:,0]]    

    # mandate equal length
    lengths = [len(data[key]) for key in keys]
    assert min(lengths) == max(lengths), 'dataset lengths unequal'

    # save out data
    for key in keys:
        torch.save(data[key], filestr+'_'+key+'.pt')

def load_expert_data(path='expert_data', loc: int = 0, track:int = 0):
    """
    Load expert data from processed files.
    Args:
        path (str): directory to save data
        loc (int): location index
        track (int): track index
    Returns:
        data (dict[torch.Tensor]): dict of data
    """
    # load observations and actions
    filestr = opj(path, intersim.LOCATIONS[loc]+'_track%03i'%(track))
    data = {}
    keys = ['ego_state', 'relative_state', 'path', 'action', 'next_ego_state', 'next_relative_state', 'next_path']
    for key in keys:
        data[key] = torch.load(filestr+'_'+key+'.pt')
    return data

def load_expert_data_raw(path='expert_data', loc: int = 0, track:int = 0):
    """
    Load expert data from raw file.
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
    obs = pickle.load(open(filestr+'_raw_observations.pkl', 'rb'))
    actions = torch.load(filestr+'_raw_actions.pt')
    actions = list(torch.unbind(actions))
    return obs, actions

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Save Expert Trajectories')
    parser.add_argument('--loc', default=0, type=int,
                       help='location (default 0)')
    parser.add_argument('--track', default=0, type=int,
                       help='track number (default 0)')
    parser.add_argument('--all-tracks', action='store_true',
                       help='whether to process all tracks at location')
    parser.add_argument('--graph', action='store_true',
                       help='whether to mask the relative states based on a ConeVisibilityGraph')
    parser.add_argument('--reg', action='store_true',
                       help='whether to regularize actions in the action targeter')
    parser.add_argument('-o', default='./expert_data', type=str,
                       help='output folder')
    args = parser.parse_args()

    kwargs = {
        'loc':args.loc, 
        'track': args.track,
        'path':args.o, 
        'mask_relstate':args.graph, 
        'regularize_actions': args.reg
    }

    if args.all_tracks:
        for i in range(intersim.MAX_TRACKS):
            kwargs['track'] = i
            generate_expert_data(**kwargs)
    else:
        generate_expert_data(**kwargs)