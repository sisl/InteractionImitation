# %%
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger

from intersim.envs.intersimple import IntersimpleReward

from gail.discriminator import MlpDiscriminator

model_name = 'gail_flat'

# %%
# Load pickled test demonstrations.
with open("data/NormalizedIntersimpleExpert_IntersimpleRewardAgent51.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# %%
# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

venv = make_vec_env(IntersimpleReward, n_envs=2, env_kwargs={'agent': 51})

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
logger.configure(tempdir_path / "GAIL/")
gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=220,
    #n_disc_updates_per_round=32,
    discrim_kwargs={'discrim_net': MlpDiscriminator()},
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=4096),
)
gail_trainer.train(total_timesteps=80000)
gail_trainer.gen_algo.save(model_name)

#del gail_trainer

# %%
model = sb3.PPO.load(model_name)

env = IntersimpleReward(agent=51)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)
