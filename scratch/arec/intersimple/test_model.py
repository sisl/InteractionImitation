from tqdm import tqdm
from copy import deepcopy
import stable_baselines3 as sb3
import intersim
from stable_baselines3.common.base_class import BaseAlgorithm
from src.baselines import IDMRulePolicy
from src.evaluation import IntersimpleEvaluation
import src.options.envs as options_envs

from typing import Optional, List, Dict, Tuple
import torch
import numpy as np

def load_policy(policy_file:str, method:str, 
    policy_kwargs:dict, skip_load:bool=False) -> Optional[BaseAlgorithm]:
    """
    Load a model given a path and the method
    
    Args:
        load_policy (str): the path to the model
        method (str): the method for the model
        skip_load (bool): whether to skip loading
    Returns:
        policy (Optional[BaseAlgorithm]): the policy to evaluate
    """
    if skip_load:
        return None
    elif method == 'idm':
        policy = IDMRulePolicy(policy_kwargs)
    elif method == 'bc':
        raise NotImplementedError
    elif method == 'gail':
        policy = sb3.PPO.load(policy_file)
        raise NotImplementedError
    elif method == 'rail':
        raise NotImplementedError
    elif method == 'sgail':
        policy = sb3.PPO.load(policy_file)
        raise NotImplementedError
    else:
        raise NotImplementedError
    return policy

def form_expert_metrics(states:torch.Tensor, actions:torch.Tensor) ->Dict[str, list]:
    """
    Given experts of tensor states and actions, form a dictionary of metrics

    Args:
        states (torch.tensor): (T+1, nv, 5) expert states for track file
        actions (torch.tensor): (T, nv, 1) expert actions for track file
    
    Returns:
        metrics (dict): dictionary maping strings to lists
    """

    T1, nv, _ = states.shape
    T, nv2, _ = actions.shape
    assert(nv==nv2)
    assert(T1==T+1)
    states = states[:T]

    # make sure metric keys and calculations match that in src.evaluation.IntersimpleEvaluation
    
    hard_brake = -3.
    timestep = 0.1
    keys = ['col_all','v_all', 'a_all','j_all', 'v_avg', 'a_avg', 'col', 'brake', 't']
    metrics = {key:[None]*nv for key in keys}
    
    import pdb
    pdb.set_trace()
    
    for i in range(nv):
        
        nni = ~torch.isnan(states[:,i,0])

        metrics['col_all'][i] = [False] * sum(nni) 
        metrics['v_all'][i] = states[nni,i,2].numpy()
        metrics['a_all'][i] = actions[nni,i,0].numpy()
        
        # jerk
        metrics['j_all'][i] = np.diff(metrics['a_all'][i]) / timestep

        # average velocity and acceleration
        metrics['v_avg'][i] = np.mean(metrics['v_all'][i])
        metrics['a_avg'][i] = np.mean(metrics['a_all'][i])

        # collision?
        metrics['col'][i] = any(metrics['col_all'][i]) 

        # brake?
        metrics['brake'][i] = any(metrics['a_all'][i] < hard_brake)

        # time length 
        metrics['t'][i] = sum(nni) 

    for key in metrics.keys():
        assert(len(metrics[key])==nv)
    return metrics

def generate_expert_metrics(locations: List[Tuple[int,int]]) -> List[Dict[str, list]]:
    """"
    Given a list of locations, for and return a list of metrics for each location

    Args:
        locations (list): list of (roundabout, track) ints

    Returns:
        expert_metrics (list of dicts): expert_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith expert roundabout trackfile
    """
    expert_metrics = []
    for (roundabout, track) in locations:
        states, actions = load_expert_states(roundabout, track)
        expert_metrics.append(form_expert_metrics(states, actions))
    return expert_metrics


def load_expert_states(roundabout:int, track:int):
    """
    Load expert states from roundabout/track info
    Args:
        roundabout (int): roundabout index
        track (int): track id
    Returns:
        states (torch.tensor): (T+1, nv, 5) expert states for track file
        actions (torch.tensor): (T, nv, 1) expert actions for track file
    """
    rname = intersim.LOCATIONS[roundabout]
    import pdb
    pdb.set_trace()
    state_path = 'expert_data/%s/track%04i/joint_expert_states.pt'%(rname, track)
    action_path = 'expert_data/%s/track%04i/joint_expert_actions.pt'%(rname, track)
    states = torch.load(state_path)
    actions = torch.load(action_path)
    return states, actions

def evaluate_policy(policy:BaseAlgorithm, locations:List[Tuple[int,int]], 
    env_class:str, env_kwargs:dict) -> List[Dict[str,list]]:
    """
    Evaluate policy on an incrementing agent environment at all locations. 
    Return metrics for that policy

    Args:
        policy (BaseAlgorithm): policy to evaluate
        locations (list of tuples): list of locations to evaluate policy
        env_class (str): name of environment to evaluate policy with
        env_kwargs (dict): key word arguments to initialize environment with
    
    Returns:
        policy_metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
    """
    envs_dict = deepcopy(intersim.envs.intersimple.__dict__)
    envs_dict.update(deepcopy(options_envs.__dict__))
    
    policy_metrics = [None]* len(locations)

    # iterate through vehicles
    for i, location in tqdm(enumerate(locations)):
        
        # add roundabout and track to environent 
        iround, track = location
        rname = intersim.LOCATIONS[iround]
        it_env_kwargs = deepcopy(env_kwargs)
        loc_kwargs = {
            'loc':iround,
            'track':track
        }
        it_env_kwargs.update(loc_kwargs)

        # initialize environment
        Env = envs_dict[env_class]
        eval_env = Env(**env_kwargs)
        evaluator = IntersimpleEvaluation(eval_env)
        policy_metrics[i] = evaluator.evaluate(policy)
    
    return policy_metrics
        


def summary_metrics(metrics:List[Dict[str,list]]):
    """
    Summarize and print metrics averaged over vehicles and roundabouts

    Args:
        metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
    """
    pass

def comparison_metrics(policy_metrics:List[Dict[str,list]], expert_metrics:List[Dict[str,list]]):
    """
    Provide distributional comparison between different sets of metrics

    Args:
        policy_metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
        expert_metrics (list of dicts): expert_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith expert roundabout trackfile
    
    """
    pass

def test_model(
    locations: List[Tuple[int,int]]= [(0,0)],
    method: str='expert', 
    policy_file: str='', 
    policy_kwargs: dict={},
    env: str='NRasterizedRouteIncrementingAgent', 
    env_kwargs: dict={},
    seed: int=0):
    """
    Test a particular model at different testing locations/tracks and compute average metrics 
    over all files. 

    Args:
        locations (list of tuples): list of (roundabout, track) integer pair testing locations
        method (str): method string
        policy_file (str): path to saved policy
        env (str): environment class
        method (str): method (expert, bc, gail, rail, hgail, hrail)
    """
    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load policy
    policy = load_policy(policy_file, method, policy_kwargs, skip_load=(method=='expert'))
    
    # load expert metrics
    expert_metrics = generate_expert_metrics(locations)

    # if we have a policy
    if policy:

        # evaluate it on the given roundabouts
        policy_metrics = evaluate_policy(policy, locations, env, env_kwargs)

        # 
        summary_metrics(policy_metrics)
        comparison_metrics(policy_metrics, expert_metrics)
    
    else:
        # if no policy, only generate summary metrics for the expert
        summary_metrics(expert_metrics)

if __name__=='__main__':
    import fire
    fire.Fire()