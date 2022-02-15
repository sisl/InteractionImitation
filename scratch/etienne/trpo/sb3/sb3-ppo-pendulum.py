from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Pendulum-v0", n_envs=4)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)
