# %%
from collections import deque
import sys
sys.path.append('../../../')

from src.discriminator import CnnDiscriminator, CnnDiscriminatorFlatAction
from imitation.algorithms import adversarial
import stable_baselines3
import pickle
import imitation.data.rollout as rollout
import tempfile
import pathlib
from imitation.util import logger
from tqdm import tqdm
from src.policies.options import OptionsCnnPolicy
from src.gail.train import flatten_transitions
from gail.options2 import OptionsEnv, RenderOptions, imitation_discriminator
from gail.envs import TLNRasterizedRouteRandomAgentLocation
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import torch
import numpy as np

model_name = 'gail_options_image_random_location'
env_settings = {'width': 70, 'height': 70, 'm_per_px': 1, 'mu': 0.001, 'random_skip': True, 'max_episode_steps': 200}

ALL_OPTIONS = [(v,t) for v in [0,2,4,8,10] for t in [5, 10, 20]] # option 0 is safe fallback

class NoisyDiscriminator(CnnDiscriminatorFlatAction):

    def __init__(self, *args, std=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.std = std

    def forward(self, state, action):
        noise = self.std * torch.randn(*action.shape, device=action.device)
        return super().forward(state, action + noise)

class LLBuffer(deque):

    def sample(self, n):
        assert n <= self.maxlen, f'Sample size of {n} exceeds buffer capacity of {self.maxlen}'
        assert n <= len(self), f'Sample size of {n} exceeds buffer size of {len(self)}'
        ind = np.random.randint(len(self), size=n)
        return list(self[i] for i in ind)

def train(
        expert_data,
        expert_batch_size=3072,
        discriminator_updates_per_round=20,
        generator_steps=1024,
        generator_batch_size=1024,
        generator_total_steps=4096,
        generator_updates_per_round=10,
        discount=1.0,
        epochs=200,
    ):
    env = TLNRasterizedRouteRandomAgentLocation(**env_settings)

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = DummyVecEnv([lambda: env])
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        #discrim_kwargs={'discrim_net': NoisyDiscriminator(venv, std=0.25)},
        disc_opt_cls=torch.optim.RMSprop,
        disc_opt_kwargs={'lr': 0.0001, 'weight_decay': 0.003},
        discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
        venv=venv, # unused
        gen_algo=stable_baselines3.PPO("CnnPolicy", venv), # unused
    )

    ll_buffer = LLBuffer(maxlen=expert_batch_size*10)

    options_env = make_vec_env(
        OptionsEnv,
        n_envs=1,
        #vec_env_cls=SubprocVecEnv,
        env_kwargs={
            'env': env,
            'options': ALL_OPTIONS,
            'discriminator': imitation_discriminator(discriminator),
            'discount': discount,
            'll_buffer': ll_buffer,
        }
    )

    generator = stable_baselines3.PPO(
        OptionsCnnPolicy,
        options_env,
        verbose=1,
        batch_size=generator_batch_size,
        n_steps=generator_steps,
        n_epochs=generator_updates_per_round,
        gamma=1.0,
        learning_rate=1e-4,
    )

    for _ in tqdm(range(epochs)):
        ll_buffer.clear()

        # train generator
        generator.learn(total_timesteps=generator_total_steps)

        # train discriminator
        for _ in range(discriminator_updates_per_round):
            generator_samples = ll_buffer.sample(expert_batch_size)
            generator_samples = flatten_transitions(generator_samples)
            discriminator.train_disc(gen_samples=generator_samples)

        generator.save(model_name)
    
    return generator

def video(model_name, env):
    model = stable_baselines3.PPO.load(model_name)

    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    env.close(filestr='render/'+model_name)

def evaluate():
    video_settings = { **env_settings, 'random_skip': False, 'max_episode_steps': 200 }
    env = TLNRasterizedRouteRandomAgentLocation(**video_settings)
    env = RenderOptions(env, options=ALL_OPTIONS)
    video(
        model_name=model_name,
        env=env
    )

# %%
if __name__ == '__main__':
    with open("data/NormalizedIntersimpleExpertMu.001N100000_TLNRasterizedRouteRandomAgentLocationw70h70mppx1mu.001rskips50.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    train(transitions)
