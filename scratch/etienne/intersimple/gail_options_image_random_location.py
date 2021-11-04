# %%
import sys
sys.path.append('../../../')

from src.discriminator import CnnDiscriminatorFlatAction
from imitation.algorithms import adversarial
import stable_baselines3
from intersim.envs import NRasterizedRouteRandomAgentLocation
import pickle
import imitation.data.rollout as rollout
import tempfile
import pathlib
from imitation.util import logger
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
from src.policies.options import OptionsCnnPolicy
from src.gail.train import flatten_transitions
from gail.options2 import OptionsEnv, RenderOptions, imitation_discriminator
from gail.envs import TLNRasterizedRouteRandomAgentLocation
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

model_name = 'gail_options_image_random_location'
env_settings = {'width': 70, 'height': 70, 'm_per_px': 1, 'mu': 0.001, 'random_skip': True, 'max_episode_steps': 50}
env = TLNRasterizedRouteRandomAgentLocation(**env_settings)

ALL_OPTIONS = [(v,t) for v in [0,2,4,6,8] for t in [5, 10, 20]] # option 0 is safe fallback

def train(
        expert_data,
        expert_batch_size=2048,
        discriminator_updates_per_round=10,
        generator_steps=256,
        generator_total_steps=1024,
        generator_updates_per_round=10,
        discount=0.99,
        epochs=100,
    ):

    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = pathlib.Path(tempdir.name)
    logger.configure(tempdir_path / "GAIL/")
    print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

    venv = DummyVecEnv([lambda: env])
    discriminator = adversarial.GAIL(
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        discrim_kwargs={'discrim_net': CnnDiscriminatorFlatAction(venv)},
        #discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
        venv=venv, # unused
        gen_algo=stable_baselines3.PPO("CnnPolicy", venv), # unused
    )

    options_env = OptionsEnv(
        env,
        options=ALL_OPTIONS,
        discriminator=imitation_discriminator(discriminator),
        discount=discount,
        ll_buffer_capacity=expert_batch_size,
    )
    generator = stable_baselines3.PPO(
        OptionsCnnPolicy,
        options_env,
        verbose=1,
        n_steps=generator_steps,
        n_epochs=generator_updates_per_round,
    )

    for _ in tqdm(range(epochs)):
        # train generator
        generator.learn(total_timesteps=generator_total_steps)

        # train discriminator
        for _ in range(discriminator_updates_per_round):
            generator_samples = options_env.sample_ll(expert_batch_size)
            generator_samples = flatten_transitions(generator_samples)
            discriminator.train_disc(gen_samples=generator_samples)

        generator.save(model_name)
    
    return generator

def video(model_name, env):
    env = RenderOptions(env, options=ALL_OPTIONS)
    model = stable_baselines3.PPO.load(model_name)

    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

    env.close(filestr='render/'+model_name)

def evaluate():
    video(
        model_name=model_name,
        env=env
    )

# %%
if __name__ == '__main__':
    with open("data/NormalizedIntersimpleExpertMu.001N10000_TLNRasterizedRouteRandomAgentLocationw70h70mppx1mu.001rskips50.pkl", "rb") as f:
        trajectories = pickle.load(f)
    transitions = rollout.flatten_trajectories(trajectories)
    train(transitions)
