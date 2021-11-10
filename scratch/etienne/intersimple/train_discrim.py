# %%
import sys
sys.path.append('../../../')

import pickle
import imitation.data.rollout as rollout
import imitation.data.types as types
import torch
from gail.envs import TLNRasterizedRouteRandomAgentLocation
import tempfile
import pathlib
from imitation.util import logger
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from imitation.algorithms import adversarial
from src.discriminator import CnnDiscriminator
import stable_baselines3
from tqdm import tqdm

with open("data/NormalizedIntersimpleExpertMu.001N50000_TLNRasterizedRouteRandomAgentLocationw70h70mppx1mu.001rskips50.pkl", "rb") as f:
    trajectories = pickle.load(f)
transitions = rollout.flatten_trajectories(trajectories)

# %%
env_settings = {'width': 70, 'height': 70, 'm_per_px': 1, 'mu': 0.001, 'random_skip': True, 'max_episode_steps': 200}
env = TLNRasterizedRouteRandomAgentLocation(**env_settings)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
logger.configure(tempdir_path / "GAIL/")
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

expert_batch_size = 4096

venv = DummyVecEnv([lambda: env])
discriminator = adversarial.GAIL(
    expert_data=transitions,
    expert_batch_size=expert_batch_size,
    #discrim_kwargs={'discrim_net': NoisyDiscriminator(venv, std=0.25)},
    disc_opt_cls=torch.optim.RMSprop,
    disc_opt_kwargs={'lr': 0.0001, 'weight_decay': 0.003},
    discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
    venv=venv, # unused
    gen_algo=stable_baselines3.PPO("CnnPolicy", venv), # unused
)

expert_data_loader = torch.utils.data.DataLoader(
    transitions,
    batch_size=expert_batch_size,
    collate_fn=types.transitions_collate_fn,
    shuffle=True,
    drop_last=True,
)

gen_data_loader = torch.utils.data.DataLoader(
    transitions,
    batch_size=expert_batch_size,
    collate_fn=types.transitions_collate_fn,
    shuffle=True,
    drop_last=True,
)

# %%
epochs = 1000
for i in tqdm(range(epochs)):
    for expert_samples, gen_samples in zip(expert_data_loader, gen_data_loader):
        # randomly corrupt actions
        gen_samples['acts'] = -1 + 2 * torch.rand(*gen_samples['acts'].shape)

        discriminator.train_disc(expert_samples=expert_samples, gen_samples=gen_samples)

    torch.save(discriminator.discrim_net.state_dict(), 'train_discrim.pt')
