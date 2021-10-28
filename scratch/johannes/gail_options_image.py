# %%
# import sys
# sys.path.append('../../../')

from src.discriminator import CnnDiscriminator, CnnDiscriminatorFlatAction
from imitation.algorithms import adversarial
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
import torch.utils.data
import numpy as np
from intersim.envs.intersimple import Intersimple, NormalizedActionSpace, NRasterized, NRasterizedInfo, NRasterizedIncrementingAgent, NRasterizedRandomAgent, speed_reward
import itertools
import functools
from torch.distributions import Categorical
import gym
import torch
import pickle
import imitation.data.rollout as rollout
import tempfile
import pathlib
from imitation.util import logger
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
from src.policies.options import OptionsCnnPolicy
from src.gail.options import OptionsEnv, LLOptions, HLOptions, RenderOptions
from src.gail.train import train_discriminator, train_generator
from src.metrics import nanmean, divergence, visualize_distribution

model_name = 'gail_options_image'
Env = NRasterized
env_settings = {'width': 36, 'height': 36, 'm_per_px': 2}

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5]] # option 0 is safe fallback

def train(expert_data, epochs=20, expert_batch_size=32, generator_steps=32, discount=0.99):
    env = Env(**env_settings)
    env.discount = discount

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(NRasterized, n_envs=1, env_kwargs=env_settings)
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        discrim_kwargs={'discrim_net': CnnDiscriminatorFlatAction(venv)},
        #discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
        venv=venv, # unused
        gen_algo=stable_baselines3.PPO("CnnPolicy", venv), # unused
    )

    generator = stable_baselines3.PPO(
        OptionsCnnPolicy,
        OptionsEnv(env, options=ALL_OPTIONS),
        verbose=1,
        n_steps=generator_steps,
    )

    # PPO.train requires logger as set up in
    # PPO._setup_learn (called by PPO.learn)
    generator._logger = stable_baselines3.common.utils.configure_logger(
        generator.verbose,
        generator.tensorboard_log,
    )

    for epoch in tqdm(range(epochs)):
        train_discriminator(LLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=expert_batch_size)
        train_generator(HLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=generator_steps)

        eval_env = Env(reward=functools.partial(speed_reward, collision_penalty=0.), **env_settings)
        ev = Evaluation(eval_env, n_eval_episodes=10)
        ev.evaluate(epoch, generator, discriminator, expert_data)
    
    return generator

from stable_baselines3.common.vec_env import VecEnv
class Evaluation:
    def __init__(self, eval_env, n_eval_episodes=10):
        # if env is a VecEnv, the code needs to be adapted, since the callback will be called after each step, 
        # so transitions of different envs will be mixed and the total number of episodes could be larger than n_eval_episodes!
        assert not isinstance(eval_env, VecEnv)
        self.env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.reset()

    def reset(self):
        self._n_collisions = 0
        self._trajectories = []
        self._episode_done = True
        self._accelerations = []

    def evaluate(self, epoch, generator, discriminator, expert_data):
        self.reset()
        metrics = {}
        
        episode_rewards, episode_lengths = evaluate_policy(
            generator, 
            self.env,
            n_eval_episodes=self.n_eval_episodes,
            callback=self.evaluate_policy_callback, 
            return_episode_rewards=True
        )

        collision_rate = self._n_collisions / self.n_eval_episodes
        metrics['collision_rate'] = collision_rate

        assert len(self._trajectories) >= self.n_eval_episodes

        # average velocity of each episode
        # this first averages velocity over single trajectories and then averages over trajectories
        # avg_velocities = [nanmean(torch.stack(t)[:,2]) for t in self._trajectories]
        # avg_velocity = np.mean(avg_velocities)
        
        # velocities produced by generator
        policy_velocities = torch.cat([torch.stack(t)[:,2] for t in self._trajectories])
        # if episodes terminate without collisions, then the state is fully nan
        policy_velocities = policy_velocities[~torch.isnan(policy_velocities)]

        # expert velocities
        extract_state = lambda info: info['projected_state'][info['agent']]
        expert_velocities = torch.stack([extract_state(info) for info in expert_data.infos])[:,2]
        expert_velocities = expert_velocities[~torch.isnan(expert_velocities)]

        metrics['avg_velocity_loss'] = (expert_velocities.mean() - policy_velocities.mean()).item()
        metrics['velocity_divergence'] = divergence(policy_velocities, expert_velocities, type='js')
        

        # accelerations produced by generator
        policy_accelerations = torch.tensor(self._accelerations)
        # expert accelerations
        extract_accel = lambda info: info['action_taken'][info['agent']]
        expert_accelerations = torch.cat([extract_accel(info) for info in expert_data.infos])

        metrics['acceleration_divergence'] = divergence(policy_accelerations, expert_accelerations, type='js')
        visualize_distribution(expert_accelerations, policy_accelerations, 'output/_action_viz{:02}'.format(epoch)) 

        print(metrics)
        return metrics

    def evaluate_policy_callback(self, local_vars, global_vars):
        venv_i = local_vars['i']
        info = local_vars['info']
        done = local_vars['done']
        _agent = info['agent']
        env = local_vars['env'].envs[venv_i]
        assert isinstance(env, Intersimple)

        # Increase collision counter if episode terminated with a collision
        if info['collision']:
            assert done
            self._n_collisions += 1

        # if last episode is done, start new trajectory
        if self._episode_done:
            self._trajectories.append([])
        agent_state = info['projected_state'][_agent]
        self._trajectories[-1].append(agent_state)
        # this does not work since agent_action are high level options
        # agent_action = local_vars['actions'][venv_i] # normalized intersimple action
        # acceleration = env._unnormalize(agent_action) if isinstance(env, NormalizedActionSpace) else agent_action
        self._accelerations.append(info['action_taken'][_agent])
        self._episode_done = done

        


# class CAPolicy:
#     def __init__(self, a):
#         self.a = torch.tensor([a])

#     def predict(self, obs, state=None, deterministic=False):
#         return self.a, state


# %%
if __name__ == '__main__':
    # %%
 
    with open("scratch/etienne/intersimple/data/NormalizedIntersimpleExpertMu.001N200_NRasterizedInfoAgent51w36h36mppx2.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)

###
    # env = NRasterizedIncrementingAgent(reward=functools.partial(speed_reward, collision_penalty=0.), **env_settings)
    # generator = CAPolicy(.5)

    # ev = Evaluation(env, 10)
    # ev.evaluate(1, generator, None, transitions)

    # exit()
###

    generator = train(transitions)

    generator.save(model_name)

    # %%
    model = stable_baselines3.PPO.load(model_name)

    env = RenderOptions(NRasterizedRandomAgent(**env_settings), options=ALL_OPTIONS)

    for s in env.sample_ll(model):
        if s['dones']:
            break

    env.close(filestr='render/'+model_name)
