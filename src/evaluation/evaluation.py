import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from intersim.envs.intersimple import Intersimple, IncrementingAgent
from typing import Callable, Dict, Optional
import os
import pickle
from tqdm import tqdm
from src.util.wrappers import IntersimpleTimeLimit
from src.options.envs import OptionsEnv
from src.safe_options.options import SafeOptionsEnv
from src.evaluation.vec_env import CallbackWhenDoneVecEnv
import matplotlib.pyplot as plt

class IntersimpleEvaluation:
    """
    Class to evaluate a policy on n_agents in an intersimple environment and store metrics for each single agent:
        - all velocities
        - all accelerations
        - all jerks
        - average velocity
        - average acceleration
        - existence time
        - whether there was a collision
        - whether there was a hard brake
    """
    def __init__(self, eval_env:IncrementingAgent, use_pbar:bool=True):
        """
        Initialize evaluation environment with an Intersimple IncrementingAgent environment

        Args:
            eval_env (IncrementingAgent): evaluation environment that increments agent upon reset
            use_pbar (bool): whether to use a progress bar
        """
        # if env is a VecEnv, the code needs to be adapted, since the callback will be called after each step, 
        # so transitions of different envs will be mixed and the total number of episodes could be larger than n_eval_episodes!
        assert not isinstance(eval_env, VecEnv)
  
        self.env = eval_env
        self.n_episodes = eval_env.nv
        self.use_pbar = use_pbar
        if isinstance(self.env, IntersimpleTimeLimit):
            self.is_options_env = isinstance(self.env.env, (OptionsEnv, SafeOptionsEnv))
        else:
            self.is_options_env = isinstance(self.env, (OptionsEnv, SafeOptionsEnv))

        # metrics present on every step of every episode
        self.metric_keys_all = ['x_all', 'y_all', 'v_all', 'a_all', 'col_all']

        # metrics calculated after the fact, with one per episode
        self.metric_keys_single = ['j_all', 'v_avg','a_avg', 'col','brake', 't']
        
        # numbers for calculating metrics
        self.hard_brake = -3. # acceleration for 'hard brake'

        # reset metrics
        self.reset()

    def reset(self):
        """
        Reset metrics prior to evaluation
        """
        self._metrics = {key: [[] for _ in range(self.n_episodes)] for key in self.metric_keys_all}
        self._metrics.update({key: [None]*self.n_episodes for key in self.metric_keys_single})
    
    def save(self, filestr):
        """
        Save metrics to filestr

        Args:
            filestr (str): path-like string to dump metrics to
        """
        # assert metrics all have correct length
        for key in self.metric_keys:
            assert len(self._metrics[key])==self.n_episodes, \
                f'_metrics[{key}] does not have length {self.n_episodes}'

        # make filepath
        os.makedirs(os.path.dirname(filestr))

        # pickle dump
        with open(filestr, 'wb') as f:
            pickle.dump(self._metrics, f)     

    def evaluate(self, policy, filestr: Optional[str] = None, videos_folder: Optional[str] = None) -> Dict[str, list]:
        """
        Evaluate a policy on the incrementing agent evaluation environment
        
        Args:
            policy (BaseClass.BaseAlgorithm): policy in which policy.predict(observation)[0] returns an action
            filestr (str): path-like string to dump metrics to or None
        """
        self.videos_folder = videos_folder

        self.reset()
        if self.use_pbar:
            self.pbar = tqdm(total=self.n_episodes)

        if self.is_options_env:
            print('Evaluating an options environment')

        evaluate_policy(
            policy, 
            self.env if self.videos_folder is None else CallbackWhenDoneVecEnv([lambda: self.env], self.done_callback),
            n_eval_episodes=self.n_episodes,
            callback=self.evaluate_options_policy_callback if self.is_options_env else self.evaluate_policy_callback,
            return_episode_rewards=False,
            render=self.videos_folder is not None,
        )
        if self.use_pbar:
            self.pbar.close()

        self.post_proc()
        if filestr:
            self.save(filestr)
        return self._metrics

    def evaluate_options_policy_callback(self, local_vars, global_vars):
        infos = local_vars['info']['ll']['infos']
        dones = local_vars['info']['ll']['env_done']
        agents = [info['agent'] for info in infos]
        for info, done, agent in zip(infos, dones, agents):
            self.eval_policy_step(info, done, agent)

    def evaluate_policy_callback(self, local_vars, global_vars):
        """
        Callback run in evaluate_policy after taking an action and receiving an observation
        
        """
        venv_i = local_vars['i']
        info = local_vars['info']
        done = local_vars['done']
        _agent = info['agent']
        env = local_vars['env'].envs[venv_i]
        # assert isinstance(env, Intersimple)

        self.eval_policy_step(info, done, _agent)

    def eval_policy_step(self, info, done, _agent):
        # Increase collision counter if episode terminated with a collision
        self._metrics['x_all'][_agent].append(info['prev_state'][_agent,0].item())
        self._metrics['y_all'][_agent].append(info['prev_state'][_agent,1].item())
        self._metrics['v_all'][_agent].append(info['prev_state'][_agent,2].item())
        self._metrics['a_all'][_agent].append(info['action_taken'][_agent,0].item())
        col = info['collision']

        # if col: # commenting out if we dont want to end on collision
        #     assert done
        self._metrics['col_all'][_agent].append(col)
        
        if done and self.use_pbar:
            self.pbar.update(1)
        
    def done_callback(self, info):
        if self.is_options_env:
            info = info['ll']['infos'][0]
        agent = info['agent']
        filestr = os.path.join(self.videos_folder, f'agent{agent}')
        self.env.close(filestr=filestr)
        plt.close('all')
    
    def post_proc(self):
        """
        Postprocess and metrics after simulation episodes
        """
        # self.metric_keys_all = ['x_all','y_all','v_all', 'a_all', 'col_all']
        # self.metric_keys_single = ['j_all', 'v_avg','a_avg', 'col','brake', 't']

        for i in range(self.n_episodes):
            self._metrics['x_all'][i] = np.array(self._metrics['x_all'][i])
            self._metrics['y_all'][i] = np.array(self._metrics['y_all'][i])
            self._metrics['v_all'][i] = np.array(self._metrics['v_all'][i])
            self._metrics['a_all'][i] = np.array(self._metrics['a_all'][i])

            # jerk
            self._metrics['j_all'][i] = np.diff(self._metrics['a_all'][i]) / self.env._env._dt

            # average velocity and acceleration
            self._metrics['v_avg'][i] = np.mean(self._metrics['v_all'][i])
            self._metrics['a_avg'][i] = np.mean(self._metrics['a_all'][i])

            # collision?
            self._metrics['col'][i] = any(self._metrics['col_all'][i]) 

            # brake?
            self._metrics['brake'][i] = any(self._metrics['a_all'][i] < self.hard_brake)

            # time length 
            self._metrics['t'][i] =len(self._metrics['v_all'][i])


def load_metrics(filestr:str) -> Dict[str,list]:
    """
    Load metrics to filestr

    Args:
        filestr (str): path-like string to dump metrics to

    Returns
        metrics (Dict[str, list])
    """
    # pickle load
    with open(filestr, 'rb') as f:
        metrics = pickle.load(f)
    return metrics  