# %%
import sys
sys.path.append('../../../')
from src.discriminator import CnnDiscriminator, CnnDiscriminatorFlatAction
from src.policies import OptionsCnnPolicy
from src.util import render_env
from src.data import load_experts
from src.gail.options import OptionsEnv, LLOptions, HLOptions, RenderOptions
from src.gail.train import train_discriminator, train_generator

from imitation.algorithms import adversarial
from imitation.util import logger
import imitation.data.rollout as rollout

import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env

import torch
import torch.utils.data
import numpy as np
import itertools
import gym
import pickle
import tempfile
import pathlib
from tqdm import tqdm

from intersim.envs.intersimple import NRasterized, NRasterizedRoute, NRasterizedRandomAgent, NRasterizedIncrementingAgent, NRasterizedRouteRandomAgent

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10]] # option 0 is safe fallback

def flatten_transitions(transitions):
    return {
        'obs': np.stack(list(t['obs'] for t in transitions), axis=0),
        'next_obs': np.stack(list(t['next_obs'] for t in transitions), axis=0),
        'acts': np.stack(list(t['acts'] for t in transitions), axis=0),
        'dones': np.stack(list(t['dones'] for t in transitions), axis=0),
    }

def train(expert_data, env_class=NRasterizedRouteRandomAgent, env_settings={}, 
        epochs=10, discrim_batch_size=32, generator_steps=2048, discount=0.99):
    """
    Args: 
        expert_data: list of transitions
        env_class: environment class
        env_settings: environment settings
        epochs: number of epochs to train for
        discrim_batch_size: discriminator batch size
        generator_steps: number of steps taken in generator
        discount: discount factor
    Returns:
        generator (stable_baselines3.PPO): options policy
    """
    env = env_class(**env_settings)
    env.discount = discount

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(env_class, n_envs=1, env_kwargs=env_settings)
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=discrim_batch_size,
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

    for _ in tqdm(range(epochs)):
        train_discriminator(LLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=discrim_batch_size)
        train_generator(HLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=generator_steps)
    
    return generator

# %%
if __name__ == '__main__':
    # %%
    model_name = 'gail_options_image_mid_wcollision'
    env_class = NRasterizedRouteRandomAgent
    env_settings = {'width': 36, 'height': 36, 'm_per_px': 2, 'stop_on_collision': False}
    
    #env_class = NRasterized
    #env_settings = {'agent': 51, 'width': 36, 'height': 36, 'm_per_px': 2}
    files = ['../../../expert_data/DR_USA_Roundabout_FT/track%04i/expert.pkl'%(i) for i in range(5)]
    transitions=load_experts(files)

    generator = train(
        transitions,
        env_class=env_class,
        env_settings=env_settings,
        epochs=2, 
        discrim_batch_size=256, 
        generator_steps=10,#256, 
        discount=0.99
        )

    generator.save(model_name)

    # Render
    render_settings = {'width': 36, 'height': 36, 'm_per_px': 2, 'agent':51, 'stop_on_collision': False}
    render_env(model_name=model_name, env='NRasterizedRoute', options=True, options_list=ALL_OPTIONS,
        **render_settings)


# %% Tests

def test_ll_expert_data():
    with open("data/NormalizedIntersimpleExpertMu.001_NRasterizedAgent51w36h36mppx2.pkl", "rb") as f:
        expert_trajectories = pickle.load(f)
    expert_transitions = rollout.flatten_trajectories(expert_trajectories)

    env = LLOptions(NRasterized(agent=51, width=36, height=36, m_per_px=2))

    gen_transitions = list(itertools.islice(env.sample_ll(
        policy=stable_baselines3.PPO(
            OptionsCnnPolicy,
            OptionsEnv(env),
            verbose=1,
        )
    ), 10))
    gen_transitions = flatten_transitions(gen_transitions)

    assert expert_transitions[:10].obs.shape == gen_transitions['obs'].shape
    assert expert_transitions[:10].next_obs.shape == gen_transitions['next_obs'].shape
    assert expert_transitions[:10].acts.shape == gen_transitions['acts'].shape
    assert expert_transitions[:10].dones.shape == gen_transitions['dones'].shape

def test_ll_states():
    env = NRasterized()
    policy = stable_baselines3.PPO(
        OptionsCnnPolicy,
        OptionsEnv(env),
        verbose=1,
    )
    llenv = LLOptions(env)
    transitions = list(itertools.islice(llenv.sample_ll(policy=policy), 100))

    env2 = NRasterized()
    s2 = env2.reset()
    for i, t in enumerate(transitions):
        assert i == 0 or np.array_equal(t['obs'], transitions[i-1]['next_obs'])
        assert np.array_equal(t['obs'], s2)
        assert t['acts'].shape == (1,)

        nexts2, _, done2, _ = env2.step(t['acts'])
        assert np.array_equal(t['next_obs'], nexts2)
        assert np.array_equal(t['dones'], done2)

        if done2:
            break

        s2 = nexts2

def test_hl_transitions():
    pass
