# %%
import pathlib
import pickle
import tempfile
import os
import random
import numpy as np
import torch

# set up ray tune
import ray
from ray import tune
from ray.tune import Analysis, ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env


from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger

from intersim.envs.intersimple import NRasterizedRandomAgent, IntersimpleReward, speed_reward, NRasterized, NRasterizedRandomAgentVerbose
import functools
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import TimeLimit

from gail.discriminator import CnnDiscriminator

model_name = 'gail_image_random_ray'
env_kwargs={'width': 36, 'height': 36, 'm_per_px': 2}

# %%

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--outdir", help="result directory", default='ray')
parser.add_argument("--test", help="test run", default=False, action="store_true")
args = parser.parse_args()
outdir = args.outdir

# %%
# Load pickled test demonstrations.
with open("data/NormalizedIntersimpleExpertMu.001N10000_NRasterizedRandomAgentw36h36mppx2.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)
# Store transitions in shared ray memory
ray_transitions = ray.put(transitions)

# %%
venv = make_vec_env(NRasterizedRandomAgent, n_envs=2, env_kwargs=env_kwargs)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")
logger.configure(tempdir_path / "GAIL/")

def get_ray_config(test=False):
    if test:
        return {
            'expert_batch_size': 2,
            'ppo_n_steps': 2,
            'ppo_batch_size': 2,
            'ppo_n_epochs': 1,
            'total_timesteps': 10,
        }
    else:
        return {
            'expert_batch_size': tune.choice([2**x for x in range(6,10)]),
            'ppo_n_steps': tune.choice([2048, 3072, 4096]),
            'ppo_batch_size': tune.choice([2**x for x in range(9,13)]),
            'ppo_n_epochs': tune.choice([6,10]),
            'total_timesteps': 400_000,
        }


def ray_train(config, checkpoint_dir=None):
    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.

    discriminator = CnnDiscriminator(venv)
    if checkpoint_dir:
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'disc_checkpoint')))
        generator = sb3.PPO.load(os.path.join(checkpoint_dir, 'gen_checkpoint'))
    else:
        generator = sb3.PPO(
            "CnnPolicy", venv, verbose=0, 
            n_steps=config["ppo_n_steps"], 
            batch_size=config["ppo_batch_size"], 
            n_epochs=config["ppo_n_epochs"]
        )
    gail_trainer = adversarial.GAIL(
        venv,
        expert_data=ray.get(ray_transitions),
        expert_batch_size=config["expert_batch_size"],
        #n_disc_updates_per_round=2048,
        discrim_kwargs={'discrim_net': discriminator},
        gen_algo=generator,
        allow_variable_horizon=True,
    )
    def callback(round):
        # eval_env = NRasterized(agent=51, reward=functools.partial(speed_reward, collision_penalty=0.), **env_kwargs)
        eval_env = TimeLimit(NRasterizedRandomAgent(reward=functools.partial(speed_reward, collision_penalty=0.), **env_kwargs), max_episode_steps=1000)
        episode_rewards, episode_lengths = evaluate_policy(generator, eval_env, return_episode_rewards=True)
        tune.report(
            reward=np.mean(episode_rewards),
            length=np.mean(episode_lengths),
            training_iteration=round,
        )
        with tune.checkpoint_dir(step=round) as checkpoint_dir:
            gail_trainer.gen_algo.save(os.path.join(checkpoint_dir, 'gen_checkpoint'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'disc_checkpoint'))

    gail_trainer.train(total_timesteps=config['total_timesteps'], callback=callback)


ray_config = get_ray_config(args.test)
search = HyperOptSearch(ray_config, metric='length', mode="max",)
search = ConcurrencyLimiter(search, max_concurrent=10)
custom_scheduler = ASHAScheduler(time_attr='training_iteration', metric='length', mode="max", grace_period=15)

analysis = tune.run(
    ray_train, 
    # config=ray_config,
    search_alg=search,
    scheduler=custom_scheduler,
    local_dir=outdir,
    resources_per_trial={"cpu":10, "gpu": 0.2},
    num_samples=1 if args.test else 100,
)

del analysis

# %%
# outdir = "ray/ray_train_2021-09-20_13-33-50/ray_train_f06785b0_33_expert_batch_size=128,ppo_batch_size=1024,ppo_n_epochs=6,ppo_n_steps=2048,total_timesteps=400000_2021-09-20_15-52-05"

# %%
analysis = Analysis(outdir, default_metric="length", default_mode="max")
filepath = analysis.get_best_logdir()
print("Best ray experiment:", filepath)
config = analysis.get_best_config()
print("Best config:", config)

# %%

model = sb3.PPO.load(os.path.join(analysis.get_last_checkpoint(), 'gen_checkpoint'))

# env = NRasterized(agent=51, **env_kwargs)
env = TimeLimit(NRasterizedRandomAgent(reward=functools.partial(speed_reward, collision_penalty=0.), **env_kwargs), max_episode_steps=1000)
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.env.close(filestr='render/'+model_name)
# %%

