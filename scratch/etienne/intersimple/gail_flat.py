# %%
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger

from intersimple.intersimple import IntersimpleReward, speed_reward

from gail.discriminator import MlpDiscriminator
import numpy as np
import functools
from stable_baselines3.common.evaluation import evaluate_policy
from ray import tune
import os
import torch

model_name = 'gail_flat'

# %%
# Load pickled test demonstrations.
#with open("data/NormalizedIntersimpleExpert_IntersimpleRewardAgent51.pkl", "rb") as f:
with open("data/NormalizedIntersimpleExpert_IntersimpleRewardAgent51Mu.001.pkl", "rb") as f:
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

def training_function(config, checkpoint_dir=None):
    logger.configure(tempdir_path / "GAIL/")

    discriminator = MlpDiscriminator()
    if checkpoint_dir:
        discriminator.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'disc_checkpoint')))
        generator = sb3.PPO.load(os.path.join(checkpoint_dir, 'gen_checkpoint'))
    else:
        generator = sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=config['n_steps'])

    gail_trainer = adversarial.GAIL(
        venv,
        expert_data=transitions,
        expert_batch_size=config['expert_batch_size'],
        n_disc_updates_per_round=config['n_disc_updates_per_round'],
        discrim_kwargs={'discrim_net': MlpDiscriminator()},
        gen_algo=generator,
    )

    def callback(epoch):
        eval_env = IntersimpleReward(agent=51, reward=functools.partial(speed_reward, collision_penalty=0.))
        #sync_envs_normalization(self.training_env, self.eval_env)
        episode_rewards, episode_lengths = evaluate_policy(generator, eval_env)
        tune.report(progress=np.mean(episode_rewards))

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            gail_trainer.gen_algo.save(os.path.join(checkpoint_dir, 'gen_checkpoint'))
            torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, 'disc_checkpoint'))

    gail_trainer.train(total_timesteps=400000, callback=callback)

analysis = tune.run(
    training_function,
    config = {
        'expert_batch_size': tune.randint(1, 220), #220,
        'n_disc_updates_per_round': tune.randint(2, 100), #16,
        'n_steps': tune.randint(1, 10000), #4096,
    },
    resources_per_trial={
        'gpu': 1,
    }
)

print('Best config', analysis.get_best_config(metric='progress', mode='max'))

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
