from stable_baselines3 import PPO
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

model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-4,
    verbose=1,
    use_sde=False,
    sde_sample_freq=4,
)
model.learn(total_timesteps=100000)
model.save('sb3-ppo-intersimple')
