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

model_name = 'bc_flat'

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

# Train BC on expert data.
# BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
logger.configure(tempdir_path / "BC/")
bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
bc_trainer.train(n_epochs=1000)
bc_trainer.save_policy(model_name)

del bc_trainer

# %%
model = bc.reconstruct_policy(model_name)

env = IntersimpleReward(agent=51)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(mode='post')
    if done:
        break

env.close(filestr='render/'+model_name)
