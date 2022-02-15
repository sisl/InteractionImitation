from sb3_contrib import TRPO
from intersim.envs import IntersimpleLidarFlat
from intersim.envs.intersimple import speed_reward
import functools

env = IntersimpleLidarFlat(
    n_rays=5,
    agent=51,
    reward=functools.partial(
        speed_reward,
        collision_penalty=0
    ),
)

model = TRPO("MlpPolicy", env, use_sde=False, sde_sample_freq=4, verbose=1)
model.learn(total_timesteps=250000)
