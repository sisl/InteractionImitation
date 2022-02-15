from sb3_contrib import TRPO
import gym
from gym.wrappers import TransformObservation

env = TransformObservation(gym.make('Pendulum-v0'), lambda obs: obs)

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)
