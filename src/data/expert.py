import intersim
from intersim.envs.intersimple import Intersimple
from stable_baselines3.common.policies import BasePolicy
import gym
import intersim.envs.intersimple
import pickle
from tqdm import tqdm
import imitation.data.rollout as rollout
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
import copy
import os
import numpy as np

class IntersimExpert(BasePolicy):
    
    def __init__(self, intersim_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=gym.spaces.Space(),
            action_space=gym.spaces.Space(),
            *args, **kwargs
        )
        self._intersim = intersim_env
        self._mu = mu

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        target_t = min(self._intersim._ind + 1, len(self._intersim._svt.simstate) - 1)
        target_state = self._intersim._svt.simstate[target_t]
        return self._intersim.target_state(target_state, mu=self._mu)

    def predict(self, *args, **kwargs):
        return self._action(), None

class IntersimpleExpert(BasePolicy):

    def __init__(self, intersimple_env, mu=0, *args, **kwargs):
        super().__init__(
            observation_space=intersimple_env.observation_space,
            action_space=intersimple_env.action_space,
            *args, **kwargs
        )
        self._intersimple = intersimple_env
        self._intersim_expert = IntersimExpert(intersimple_env._env, mu=mu)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def _action(self):
        return self._intersim_expert._action()[self._intersimple._agent]

    def predict(self, *args, **kwargs):
        return self._action(), None

class NormalizedIntersimpleExpert(IntersimpleExpert):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        action, _ = super().predict(*args, **kwargs)
        return self._intersimple._normalize(action), None

class DummyVecEnvPolicy(BasePolicy):

    def __init__(self, experts):
        self._experts = [e() for e in experts]

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()
    
    def predict(self, *args, **kwargs):
        predictions = [e.predict() for e in self._experts]
        actions = [p[0] for p in predictions]
        states = [p[1] for p in predictions]
        return actions, states
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

def save_video(env, expert):
    env.reset()
    env.render()
    done = False
    while not done:
        actions, _ = expert.predict()
        _, _, done, _ = env.step(actions)
        env.render()
    env.close()

class NoShuffleRNG(np.random.RandomState):
    """
    A np.random.RandomState rng that doesn't shuffle inputs (for imitation.rollout)
    """
    def __init__(self):
        super().__init__()

    def shuffle(self, x):
        return x

def load_experts(expert_files, flatten=True):
    """
    Load expert trajectories from files and combine their transitions into a single RB

    Args:
        expert_files (list): list of expert file strings
        flatten (bool): whether to flatten trajectory info
    Returns:
        transitions (list): list of combined expert episode transitions 
    """
    transitions = []
    for file in tqdm(expert_files):
        with open(file, "rb") as f:
            new_trajectories = pickle.load(f)
        transitions += new_trajectories
    if flatten:
        transitions = rollout.flatten_trajectories(transitions)
    return transitions

def single_agent_expert(expert='NormalizedIntersimpleExpert', 
                        env='NRasterizedRouteIncrementingAgent', 
                        env_args={}, policy_args={}, **kwargs):
    """
    Args: 
        expert (class): class of expert
        env (class): class of env intersim.envs.intersimple
        env_args (dict): dictionary of kwargs when instantiating environment class
        policy_args (dict): dictionary of kwargs when instantiating Expert policy
        path (str): path to store output
        min_timesteps (int): min number of timesteps for call to rollout.rollout_and_save
        min_episodes (int): min number of episodes for call to rollout.rollout_and_save
        video (bool): whether to save a video of the expert until a single environment instantiation stops
    """
    Env = intersim.envs.intersimple.__dict__[env]
    Expert = globals()[expert]
    env = Env(**env_args)
    policy = Expert(env, **policy_args)
    single_agent_demonstrations(env, policy, **kwargs)

def single_agent_demonstrations(env, policy, 
                                path=None, min_timesteps=None, 
                                min_episodes=None, video=False,
                                env_args={}, policy_args={}):
    """Rollout and save expert demos.
    
    Usage:
        python -m intersimple.expert <flags>
    Args: 
        env (class): intersimple environment
        policy (BasePolicy): intersimple policy
        path (str): path to store output
        min_timesteps (int): min number of timesteps for call to rollout.rollout_and_save
        min_episodes (int): min number of episodes for call to rollout.rollout_and_save
        video (bool): whether to save a video of the expert until a single environment instantiation stops
        env_args (dict): dictionary of kwargs when instantiating environment class
        policy_args (dict): dictionary of kwargs when instantiating Expert policy
    """
    
    info_env = RolloutInfoWrapper(env) # getting rollout info (dictionary) from environment
    venv = DummyVecEnv([lambda: info_env]) # making a DummyVecEnv with a list of a function that when called returns the rollout info
    venv_policy = DummyVecEnvPolicy([lambda: policy]) # make a DummyVecEnvPolicy with a list of a function that when called returns the Expert policy

    if min_timesteps is None and min_episodes is None:
        min_episodes = env.nv # one episode per vehicle being controlled in environment (hopefully an incrementing agent environment)

    if video:
        save_video(env, policy)

    path = path or (policy.__class__.__name__ + '_' + env.__class__.__name__ + '.pkl')
    suntil = rollout.make_sample_until(
            min_timesteps=min_timesteps,
            min_episodes=min_episodes,
        )
    
    rollout.rollout_and_save(
        path=path,
        policy=venv_policy,
        venv=venv,
        sample_until=suntil,
        rng=NoShuffleRNG()
    )

def multi_agent_demonstrations(expert='IntersimExpert',path=None, env_args={}, policy_args={}):
    """
    Run and save the `intersim' multiagent environment demonstration
    
    Args: 
        expert (class): class of multi-agent expert
        path (str): path to store output data
        env_args (dict): dictionary of kwargs when instantiating environment class
        policy_args (dict): dictionary of kwargs when instantiating Expert policy
    """
    if path is None:
        raise('No path specified')

    env = gym.make('intersim:intersim-v0',**env_args)
    Expert = globals()[expert] 
    policy = Expert(env, **policy_args)

    s, done = env.reset(), False
    env.render(mode='file')
    while not done:
        _,_,done,_ = env.step(policy.predict()[0])
        env.render(mode='file')
    env.close(filestr=path)

def process_experts(filename:str='expert.pkl', 
                    locs:list=None, 
                    tracks:list=None,
                    env_class:str='NRasterizedRouteIncrementingAgent',
                    env_args:dict={'width':36,'height':36,'m_per_px':2},
                    expert_class:str='NormalizedIntersimpleExpert',  
                    expert_args:dict={'mu':0.001}):
    """
    Process all experts in the Interaction Dataset
    For now, using NormalizedIntersimpleExpert with NRasterizedRouteIncrementingAgent environment

    Args:
        filename (str): name for track file
        locs (list): list of location ids
        tracks (list): list of track numbers
        env_class (str): class of environment
        env_args (dict): default environment kwargs
        expert_class (str): class of expert
        expert_args (dict): default expert kwargs
    """
    locs = locs or intersim.LOCATIONS
    tracks = tracks or range(intersim.MAX_TRACKS)
    pbar = tqdm(total=len(locs)*len(tracks))
    for loc in locs:
        for track in tracks:
            
            iloc = intersim.LOCATIONS.index(loc)

            it_env_args = copy.deepcopy(env_args)
            env_loc_args = {
                'loc':iloc,
                'track':track,
            }
            it_env_args.update(env_loc_args)
            out_folder = os.path.join('expert_data',loc, 'track%04i'%(track))
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder) 
            it_path = os.path.join(out_folder,filename)

            # Multi-Agent demonstrations
            it_ma_path = os.path.join(out_folder,'joint_expert')
            multi_agent_demonstrations(
                expert='IntersimExpert',
                path=it_ma_path,
                env_args=env_loc_args,
                policy_args=expert_args
            )
            # Single-Agent POV Demonstrations
            single_agent_expert(
                expert=expert_class, 
                env=env_class, 
                path=it_path, 
                env_args=it_env_args, 
                policy_args=expert_args,
                )
            
            pbar.update(1)
    pbar.close()
    
if __name__=='__main__':
    import fire
    fire.Fire(process_experts)

