from tqdm import tqdm
from copy import deepcopy
import stable_baselines3 as sb3
import intersim

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10]] # option 0 is safe fallback

def load_model(model_path:str, method:str):
    """
    Load a model given a path and the method
    
    Args:
        model_path (str): the path to the model
        method (str): the method for the model
    Returns:
        model: the action model
        is_heir (bool): whether the method is heirarchial
    """
    model = None
    is_heir = False
    if method == 'expert':
        raise NotImplementedError
    elif method == 'bc':
        raise NotImplementedError
    elif method == 'gail':
        raise NotImplementedError
    elif method == 'rail':
        raise NotImplementedError
    elif method == 'hgail':
        is_heir = True
        model = sb3.PPO.load(model_path)
    elif method == 'hrail':
        is_heir = True
        raise NotImplementedError
    else:
        raise NotImplementedError
    return model, is_heir

def load_expert_states(roundabout, track):
    """
    Load expert states from roundabout/track info
    Args:
        roundabout (str): roundabout name
        track (str): track id
    Returns:
        states (torch.tensor): (T+1, nv, 5) expert states for track file
        actions (torch.tensor): (T, nv, 1) expert actions for track file
    """
    state_path = '../../../expert_data/%s/track%04i/joint_expert_states.pt'%(roundabout, track)] #FIXME when moving
    action_path = '../../../expert_data/%s/track%04i/joint_expert_actions.pt'%(roundabout, track)] #FIXME when moving
    states = torch.load(path)
    actions = torch.load(path)
    # nanify actions where vehicle's don't exist
    import pdb
    pdb.set_trace()
    return states, actions

def test_model(
    locations=[(0,0)], 
    model_name='gail_image_multiagent_nocollision', 
    env='NRasterizedRouteIncrementingAgent', 
    method='expert',
    options_list=ALL_OPTIONS, 
    **env_kwargs):
    """
    Test a particular model at different locations/tracks

    Args:
        locations (list of tuples): list of (roundabout, track) integer pairs
        model_name (str): name of model to test
        env (str): environment class
        method (str): method (expert, bc, gail, rail, hgail, hrail)
        options_list (list): list of options
    """

    # load policy
    policy, is_heir = load_model(model_name, method)

    # iterate through vehicles
    all_vehicle_infos = []
    for i, location in tqdm(enumerate(locations)):
        
        # add roundabout and track to environent 
        roundabout, track = location
        iround = intersim.LOCATIONS.index(roundabout)
        it_env_kwargs = deepcopy(env_kwargs)
        loc_kwargs = {
            'loc':iround,
            'track':track
        }
        it_env_kwargs.update(loc_kwargs)
        
        # load expert states and get average velocities
        expert_states, expert_actions = load_expert_states(roundabout, track)
        expert_vavg = torch.nanmean(expert_states[:,:,3], dim=-1)

        # initialize environment
        if not is_heir:
            Env = src.options.envs.__dict__[env]
        else: 
            Env = intersim.envs.intersimple.__dict__[env]
        env = Env(**env_kwargs)
        s = env.reset()

        # Iterate through every vehicle and time
        vehicle_infos, done = [], False
        for iv in range(env.nv):
            v_number = env.agent
            i_vehicle_infos = {'s':[], 'a':[], 'it':[]}
            while not done:
                a = policy(s)
                sp, r, done, info   = env.step(a)
                i_vehicle_infos['s'].append(env._env.state) # FIX
                i_vehicle_infos['a'].append(a) 
                i_vehicle_infos['it'].append(env._env.it) # FIX
            i_vehicle_info.update({
                'vehicle_id': env.agent,
                'n_steps': len(i_vehicle_infos['a']),
                'T': len(i_vehicle_infos['a'])*env._env.dt, # FIX
                'n_collisions': collision.check(i_vehicle_infos['s'], env._env.lengths. env._env.widths), # FIX
                'expert_vavg': expert_vavg[env.agent]
            })
            vehicle_infos.append(i_vehicle_info)
            env.reset()

        all_vehicle_infos.append({
            'loc': location,
            'track': track,
            'stats': vehicle_infos
            })
        env.close()

    # print and save model-specific metrics
    outfolder = 'test_metrics'
    print_and_save(all_vehicle_infos, method, model, outfolder)

def print_and_save(stats, method, model, outfolder):
    """
    Print and save stats
    """
    pass

def load_compare():
    pass

if __name__=='__main__':
    import fire
    fire.Fire()