# %%
# import sys
# sys.path.append('../../../')

from src.discriminator import CnnDiscriminator, CnnDiscriminatorFlatAction
from imitation.algorithms import adversarial
import stable_baselines3
from stable_baselines3.common.evaluation import evaluate_policy
import torch.utils.data
import numpy as np
from intersim.envs.intersimple import Intersimple, NRasterized, NRasterizedInfo, NRasterizedIncrementingAgent, NRasterizedRandomAgent, speed_reward
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

model_name = 'gail_options_image'
Env = NRasterizedRandomAgent
env_settings = {'width': 36, 'height': 36, 'm_per_px': 2}

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5]] # option 0 is safe fallback

def train(expert_data, epochs=20, expert_batch_size=32, generator_steps=32, discount=0.99):
    env = Env(**env_settings)
    env.discount = discount

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = make_vec_env(Env, n_envs=1, env_kwargs=env_settings)
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
