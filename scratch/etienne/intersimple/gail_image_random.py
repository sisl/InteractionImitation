# %%
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger

from intersim.envs.intersimple import NRasterizedRandomAgent, IntersimpleReward, speed_reward
import functools
from stable_baselines3.common.evaluation import evaluate_policy

from gail.discriminator import CnnDiscriminator

model_name = 'gail_image_random'

# %%
# Load pickled test demonstrations.
with open("data/NormalizedIntersimpleExpertMu.001_NRasterizedRandomAgentw36h36mppx2.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# %%
# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

env_kwargs = {'width': 36, 'height': 36, 'm_per_px': 2}
venv = make_vec_env(NRasterizedRandomAgent, n_envs=2, env_kwargs=env_kwargs)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
logger.configure(tempdir_path / "GAIL/")
generator = sb3.PPO("CnnPolicy", venv, verbose=1, n_steps=1024)
gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    #n_disc_updates_per_round=2048,
    discrim_kwargs={'discrim_net': CnnDiscriminator(venv)},
    gen_algo=generator,
    allow_variable_horizon=True,
)
def callback(round):
    eval_env = NRasterizedRandomAgent(reward=functools.partial(speed_reward, collision_penalty=0.), **env_kwargs)
    #sync_envs_normalization(self.training_env, self.eval_env)
    episode_rewards, episode_lengths = evaluate_policy(generator, eval_env, return_episode_rewards=True)

gail_trainer.train(total_timesteps=100000, callback=callback)
gail_trainer.gen_algo.save(model_name)

#del gail_trainer

# %%
model = sb3.PPO.load(model_name)

env = NRasterizedRandomAgent(width=36, height=36, m_per_px=2)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)
