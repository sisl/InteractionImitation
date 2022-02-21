from tqdm import tqdm
from copy import deepcopy
import stable_baselines3 as sb3
import intersim
from intersim.envs import Intersimple
from stable_baselines3.common.base_class import BaseAlgorithm
from src.baselines import IDMRulePolicy
from src.evaluation import IntersimpleEvaluation
import src.gail.options as options_envs
from src.evaluation.metrics import divergence, visualize_distribution, rwse
from src.evaluation.utils import save_metrics
from src.core.policy import SetPolicy, SetDiscretePolicy
from src.core.reparam_module import ReparamPolicy, ReparamSafePolicy
from src.options import envs as options_envs2
from src.safe_options.policy import SetMaskedDiscretePolicy
from src.safe_options import options as options_envs3

from typing import Optional, List, Dict, Tuple
import torch
import numpy as np

#(method, policy_file, policy_kwargs, eval_env)

def load_policy(method:str, 
    policy_file:str, 
    policy_kwargs:dict, 
    env: Intersimple) -> BaseAlgorithm:
    """
    Load a model given a path and the method
    
    Args:
        method (str): the method for the model
        policy_file (str): the path to the model
        policy_kwargs (str): the path to the model
        env (Intersimple): Intersimple environment for evaluation (necessary for IDM policy)
    Returns:
        policy (Optional[BaseAlgorithm]): the policy to evaluate
    """
    if method == 'idm':
        policy = IDMRulePolicy(env, **policy_kwargs)
    elif method == 'bc':
        policy = SetPolicy(env.action_space.shape[-1])
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'gail':
        policy = SetPolicy(env.action_space.shape[-1])
        policy(torch.zeros(env.observation_space.shape))
        policy = ReparamPolicy(policy)
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'gail-ppo':
        policy = SetPolicy(env.action_space.shape[-1])
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'rail':
        raise NotImplementedError
    elif method == 'ogail':
        policy = SetDiscretePolicy(env.action_space.n)
        policy(torch.zeros(env.observation_space.shape))
        policy = ReparamPolicy(policy)
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'ogail-ppo':
        policy = SetDiscretePolicy(env.action_space.n)
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'sgail':
        policy = SetMaskedDiscretePolicy(env.action_space.n)
        policy(
            torch.zeros(env.observation_space['observation'].shape),
            torch.zeros(env.observation_space['safe_actions'].shape)
        )
        policy = ReparamSafePolicy(policy)
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
    elif method == 'sgail-ppo':
        policy = SetMaskedDiscretePolicy(env.action_space.n)
        policy.load_state_dict(torch.load(policy_file))
        policy.eval()
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
    keys = ['col_all','x_all','y_all','v_all', 'a_all','j_all', 'v_avg', 'a_avg', 'col', 'brake', 't']
    metrics = {key:[None]*nv for key in keys}
    
    for i in range(nv):
        
        nni = ~torch.isnan(states[:,i,0])

        metrics['col_all'][i] = [False] * sum(nni) 
        metrics['x_all'][i] = states[nni,i,0].numpy()
        metrics['y_all'][i] = states[nni,i,1].numpy()
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
    state_path = 'expert_data/%s/track%04i/joint_expert_states.pt'%(rname, track)
    action_path = 'expert_data/%s/track%04i/joint_expert_actions.pt'%(rname, track)
    states = torch.load(state_path)
    actions = torch.load(action_path)
    return states, actions

def evaluate_policy(locations:List[Tuple[int,int]], 
    env_class:str, 
    env_kwargs:dict,
    method: str,
    policy_file: str,
    policy_kwargs:dict) -> List[Dict[str,list]]:
    """
    Evaluate policy on an incrementing agent environment at all locations. 
    Return metrics for that policy

    Args:
        policy (BaseAlgorithm): policy to evaluate
        locations (list of tuples): list of locations to evaluate policy
        env_class (str): name of environment to evaluate policy with
        env_kwargs (dict): key word arguments to initialize environment with
        method (str): policy method
        policy_file (str): policy file path
        policy_kwargs (dict): policy kwargs
    
    Returns:
        policy_metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
    """
    envs_dict = dict(intersim.envs.intersimple.__dict__)
    envs_dict.update(dict(options_envs.__dict__))
    envs_dict.update(dict(options_envs2.__dict__))
    envs_dict.update(dict(options_envs3.__dict__))
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

        # load policy
        policy = load_policy(method, policy_file, policy_kwargs, eval_env)

        # run policy on environment
        policy_metrics[i] = evaluator.evaluate(policy)
    
    return policy_metrics
        
def summary_metrics(metrics:List[Dict[str,list]]) -> Dict[str,float]:
    """
    Summarize and print metrics averaged over vehicles and roundabouts

    Args:
        metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
    
    Returns:
        summary_metrics (Dict[str,float]): maps summary metric descriptions to values
    """
    # keys = ['col_all','v_all', 'a_all','j_all', 'v_avg', 'a_avg', 'col', 'brake', 't']
    summary_metrics = {}

    # average average-velocity
    all_vavgs =  sum([d['v_avg'] for d in metrics],[]) # aggregate to single list
    summary_metrics['mean average velocity'] = sum(all_vavgs)/len(all_vavgs)
    
    # average acceleration
    all_aalls = np.concatenate([np.concatenate(d['a_all']) for d in metrics])
    summary_metrics['mean acceleration'] = np.mean(all_aalls)

    # average +acceleration
    pos_accels = all_aalls[all_aalls>0]
    summary_metrics['mean positive acceleration'] = np.mean(pos_accels)

    # average deceleration
    decels = all_aalls[all_aalls<0]
    summary_metrics['mean deceleration'] = np.mean(decels)

    # average jerk
    all_jerks = np.concatenate([np.concatenate(d['j_all']) for d in metrics])
    summary_metrics['mean jerk'] = np.mean(all_jerks)

    # average |jerk|
    summary_metrics['mean |jerk|'] = np.mean(np.abs(all_jerks))

    # collision rate
    all_collisions = sum([d['col'] for d in metrics],[]) # aggregate to single list
    summary_metrics['collision rate'] = sum(all_collisions)/len(all_collisions)

    # hard brake rate
    all_hard_brakes = sum([d['brake'] for d in metrics],[]) # aggregate to single list
    summary_metrics['hard brake rate'] = sum(all_hard_brakes)/len(all_hard_brakes)

    # average number of timesteps
    all_ts = sum([d['t'] for d in metrics],[]) # aggregate to single list
    summary_metrics['mean episode length'] = sum(all_ts)/len(all_ts)

    for key in summary_metrics.keys():
        print(f'{key}: {summary_metrics[key]}')

    return summary_metrics

def comparison_metrics(policy_metrics:List[Dict[str,list]], 
    expert_metrics:List[Dict[str,list]], outbase:str='' ) -> Dict[str,float]:
    """
    Provide distributional comparison between different sets of metrics

    Args:
        policy_metrics (list of dicts): policy_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith roundabout trackfile under a policy
        expert_metrics (list of dicts): expert_metrics[i][j][k] returns the value of metric 'j' 
            evaluated on the kth episode (car) of the ith expert roundabout trackfile
        outbase (str): path to save output figs to

    Returns:   
        comparison_metrics (Dict[str,float]): dict mapping comparison metric description to value
    """
    comparison_metrics = {}

    # rwse
    expert_traj, policy_traj = [], []
    for iR in range(len(policy_metrics)):
        for iTraj in range(len(policy_metrics[iR]['x_all'])):
            expert_traj.append(np.vstack((expert_metrics[iR]['x_all'][iTraj], expert_metrics[iR]['y_all'][iTraj]))) 
            policy_traj.append(np.vstack((policy_metrics[iR]['x_all'][iTraj], policy_metrics[iR]['y_all'][iTraj]))) 
    assert len(expert_traj)==len(policy_traj)
    comparison_metrics['rwse'] = rwse(expert_traj, policy_traj)

    # average velocity shortfall
    expert_vavg = np.array(sum([d['v_avg'] for d in expert_metrics],[]))
    policy_vavg = np.array(sum([d['v_avg'] for d in policy_metrics],[]))
    assert len(expert_vavg)==len(policy_vavg)
    comparison_metrics['mean shortfall velocity'] = np.mean(expert_vavg - policy_vavg)

    # Average |Delta V average|
    comparison_metrics['average absolute average velocity'] = np.mean(np.abs(expert_vavg - policy_vavg))
    
    # velocity JSD
    expert_vs = torch.tensor(np.concatenate([np.concatenate(d['v_all']) for d in expert_metrics]))
    policy_vs = torch.tensor(np.concatenate([np.concatenate(d['v_all']) for d in policy_metrics]))
    comparison_metrics['velocity distribution divergence'] = divergence(expert_vs, policy_vs)
    visualize_distribution(expert_vs, policy_vs, outbase+'_velocity_jsd')
    
    # acceleration JSD
    expert_as = torch.tensor(np.concatenate([np.concatenate(d['a_all']) for d in expert_metrics]))
    policy_as = torch.tensor(np.concatenate([np.concatenate(d['a_all']) for d in policy_metrics]))
    comparison_metrics['acceleration distribution divergence'] = divergence(expert_as, policy_as)
    visualize_distribution(expert_as, policy_as, outbase+'_accel_jsd')
    
    # jerk JSD
    expert_jerks = torch.tensor(np.concatenate([np.concatenate(d['j_all']) for d in expert_metrics]))
    policy_jerks = torch.tensor(np.concatenate([np.concatenate(d['j_all']) for d in policy_metrics]))
    comparison_metrics['jerk distribution divergence'] = divergence(expert_jerks, policy_jerks)
    visualize_distribution(expert_jerks, policy_jerks, outbase+'_jerk_jsd')

    for key in comparison_metrics.keys():
        print(f'{key}: {comparison_metrics[key]}')
        
    return comparison_metrics

def eval_main(
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
    print(f'Evaluating {method} on {env}')

    # set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    pfilename = policy_file.split('/')[-1].split('.')[0]
    outbase = f'out/{method}/{pfilename}_seed{seed}'

    # load expert metrics
    expert_metrics = generate_expert_metrics(locations)

    # no comparison for expert
    if method=='expert':
        smetrics = summary_metrics(expert_metrics)
        save_metrics(smetrics, outbase+'_summary.pkl')
    
    # otherwise evaluate policy on roundabouts and generate metrics
    else:

        # evaluate it on the given roundabouts
        policy_metrics = evaluate_policy(locations, env, env_kwargs, method, policy_file, policy_kwargs)
        smetrics = summary_metrics(policy_metrics)
        save_metrics(smetrics, outbase+'_summary.pkl')
        cmetrics = comparison_metrics(policy_metrics, expert_metrics, outbase=outbase)
        save_metrics(cmetrics, outbase+'_comparison.pkl')

if __name__=='__main__':
    import fire
    fire.Fire(eval_main)