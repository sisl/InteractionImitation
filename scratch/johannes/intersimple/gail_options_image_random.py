# %%
import sys
sys.path.append('../../../')

from src.discriminator import CnnDiscriminatorFlatAction
from imitation.algorithms import adversarial
import stable_baselines3
import torch.utils.data
import numpy as np
from intersim.envs.intersimple import NRasterizedRandomAgent
import itertools
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
from src.evaluation.evaluation import Evaluation
from torch.utils.tensorboard import SummaryWriter
import os 

model_name = 'gail_options_image_random'
env_settings = {'width': 36, 'height': 36, 'm_per_px': 2}

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10, 20]] # option 0 is safe fallback

def train(expert_data, epochs=100, expert_batch_size=16, generator_steps=16, discount=0.99):
    env = NRasterizedRandomAgent(**env_settings)
    env.discount = discount

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(NRasterizedRandomAgent, n_envs=1, env_kwargs=env_settings)
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
    
    filestr = os.path.join('out', model_name)
    writer = SummaryWriter(filestr)
    ev = Evaluation(filestr, env, expert_data, n_eval_episodes=100)
    for epoch in tqdm(range(epochs)):
        train_discriminator(LLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=expert_batch_size)
        train_generator(HLOptions(env, options=ALL_OPTIONS), generator, discriminator, num_samples=generator_steps)
        generator.save(model_name)
    
        metrics = ev.evaluate(epoch, generator, discriminator)
        for metric, value in metrics.items():
            writer.add_scalar(metric, value, epoch)
    
    return generator

def video(model_name, env):
    model = stable_baselines3.PPO.load(model_name)
    env = RenderOptions(env, options=ALL_OPTIONS)
    for s in env.sample_ll(model):
        if s['dones']:
            break
    env.close(filestr='render/'+model_name)

def evaluate():
    video(
        model_name=model_name,
        env=NRasterizedRandomAgent(**env_settings)
    )

# %%
if __name__ == '__main__':

    with open("scratch/etienne/intersimple/data/NormalizedIntersimpleExpertMu.001N10000_NRasterizedRandomAgentInfow36h36mppx2.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    train(transitions)
