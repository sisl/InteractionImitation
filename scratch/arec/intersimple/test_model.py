from tqdm import tqdm
from copy import deepcopy

ALL_OPTIONS =

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
        pass
    elif method == 'bc':
        raise NotImplementedError
    elif method == 'gail':
        raise NotImplementedError
    elif method == 'rail':
        raise NotImplementedError
    elif method == 'hgail':
        is_heir = True
        raise NotImplementedError
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
        expert_states (torch.tensor): (nv, T, 5) expert states for track file
    """
    pass

def test_model(
    locations=[], 
    model_name='gail_image_multiagent_nocollision', 
    env='NRasterizedRouteIncrementingAgent', 
    method='expert',
    options_list=ALL_OPTIONS, 
    **env_kwargs):
    """
    Test a particular model at different locations/tracks

    Args:
        locations (list of tuples): list of (roundabout, track) pairs
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
        
        # load expert states
        expert_states = load_expert_states(roundabout, track)

        # add roundabout and track to environent 
        roundabout, track = location
        it_env_kwargs = deepcopy(env_kwargs)
        it_env_kwargs.update({})
        
        # load expert states and get average velocities
        expert_states = load_expert_states(roundabout, track)
        expert_vavg = torch.nanmean(expert_states[:,:,3], dim=-1)

        # initialize environment
        if not is_heir:
            pass
        else: 
            pass
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